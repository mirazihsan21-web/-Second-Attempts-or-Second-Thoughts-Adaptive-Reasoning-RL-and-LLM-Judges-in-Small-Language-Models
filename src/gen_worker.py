# gen_worker.py: batched vLLM generation in a fresh process (clean GPU memory every call).
# Usage: python gen_worker.py job.json
# job = {"model": hf_id_or_dir, "lora": dir_or_null, "max_model_len": int, "gpu_mem": float,
#        "out": path.jsonl, "requests": [{"id", "prompt", "n", "temperature", "top_p",
#        "max_tokens", "seed", "stop"(optional)}]}
import json, sys, time
from pathlib import Path


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["out"])
    done_flag = out.with_suffix(".done.json")
    if done_flag.exists():
        print("[gen_worker] already complete:", out)
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    llm_kwargs = dict(model=job["model"], dtype="bfloat16", max_model_len=job["max_model_len"],
                      gpu_memory_utilization=job.get("gpu_mem", 0.85), seed=0,
                      enable_prefix_caching=True, trust_remote_code=True,
                      enforce_eager=bool(job.get("enforce_eager", False)),   # eager = faster start-up, less memory
                      max_num_seqs=int(job.get("max_num_seqs", 256)))        # lowered automatically after an OOM
    if job.get("lora"):
        llm_kwargs.update(enable_lora=True, max_lora_rank=job.get("max_lora_rank", 64), max_loras=1)
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    load_s = time.time() - t0

    prompts, params, meta = [], [], []
    for r in job["requests"]:
        ptoks = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        room = job["max_model_len"] - ptoks
        if room < 16:
            raise ValueError(f"Prompt {r['id']} ({ptoks} tokens) does not fit the context; "
                             "revise the protocol instead of silently dropping it.")
        max_tokens = min(int(r["max_tokens"]), room)
        params.append(SamplingParams(n=int(r.get("n", 1)), temperature=float(r.get("temperature", 0.0)),
                                     top_p=float(r.get("top_p", 1.0)), max_tokens=max_tokens,
                                     seed=r.get("seed"), stop=r.get("stop"),
                                     repetition_penalty=1.0))       # never a repetition penalty
        prompts.append(r["prompt"])
        meta.append({"prompt_tokens": ptoks, "max_tokens": max_tokens,
                     "context_capped": max_tokens < int(r["max_tokens"])})

    # The parameters are explicit, so a model's generation_config.json defaults (e.g. Qwen2.5-1.5B-Instruct:
    # repetition_penalty 1.1, top_k 20) never apply. Record what was actually used.
    sampling = {k: getattr(params[0], k, None) for k in ("temperature", "top_p", "top_k", "repetition_penalty")} if params else {}
    assert all(p.repetition_penalty == 1.0 for p in params), "a repetition penalty slipped into evaluation"

    lora_req = LoRARequest("adapter", 1, job["lora"]) if job.get("lora") else None
    t1 = time.time()
    results = llm.generate(prompts, params, lora_request=lora_req, use_tqdm=True)
    gen_s = time.time() - t1

    tmp = out.with_suffix(".tmp")
    n_out = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r, m, res in zip(job["requests"], meta, results):
            for k, c in enumerate(res.outputs):
                f.write(json.dumps({"id": r["id"], "sample": k, "text": c.text,
                                    "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                                    **m}, ensure_ascii=False) + "\n")
                n_out += 1
    tmp.replace(out)
    done_flag.write_text(json.dumps({"rows": n_out, "load_seconds": load_s, "generate_seconds": gen_s,
                                     "requests": len(prompts), "model": job["model"],
                                     "lora": job.get("lora"), "sampling_first_request": sampling},
                                    indent=2, default=str), encoding="utf-8")
    print(f"[gen_worker] {n_out} completions in {gen_s:.1f}s (load {load_s:.1f}s) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])