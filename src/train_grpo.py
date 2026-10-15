# train_grpo.py: Stage B. GRPO from the SFT-v2 adapter with verifiable (or judge) rewards.
# Usage: python train_grpo.py config.json
# Variants (cfg["variant"]):
#   ours       : strict correctness + format + repetition + difficulty-aware mode reward      (R1, C2)
#   judge      : 1.5B LLM-judge correctness (+ the same difficulty-aware bonus, judge-gated)
#                + format + repetition                                                   (R2, C3)
#   std_length : strict correctness + format + repetition + standard length bonus       (R3 ablation)
#   no_length  : strict correctness + format + repetition                               (control C1, S8)
# cfg["scale_rewards"]: TRL advantage scaling ("group" = TRL default, "batch" = R1/R3 since S6, "none" = Dr. GRPO-style).
# cfg["bonus_scale"]  : multiplies the shaping reward's weight (control C3: 0.1). TRL sums weighted rewards BEFORE
#                       the advantage normalisation, so this is exactly a smaller bonus.
# Every variant LOGS the same monitors (strict accuracy, think/short/malformed rate) with weight 0,
# so training curves are directly comparable and judge reward-hacking is visible.
# cfg["balanced"] = True (R1b): AdaptThink-style balanced think/short exploration (see make_balanced_rollout).
import json, math, os, sys, time
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (correctness_reward, format_reward, repetition_reward, mode_length_reward,
                    adaptive_mode_reward, strict_accuracy, think_rate, short_rate, malformed_rate,
                    think_correct, policy_think_prob, all_correct_group)  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------- LLM judge (R2) ---------------------------------------------
class Judge:
    def __init__(self, name, max_new_tokens, batch_size, parser="fixed"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").cuda().eval()
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        # "fixed" = the judge's actual verdict (the number completing 'SCORE:'); "v1" = v1's first-number parser,
        # which mostly read numbers from the judge's own re-solution (pilot finding). Default: a fair test of the judge.
        self.parse = common.parse_judge_fixed if parser == "fixed" else common.parse_judge_v1
        self.log = []

    @torch.no_grad()
    def score(self, problems, references, candidates):
        """Batched greedy judging. Adaptive: on CUDA OOM the batch is halved (and remembered) instead of crashing."""
        out, i = [], 0
        while i < len(candidates):
            bs = self.batch_size
            prompts = [common.judge_prompt(p, r, c) for p, r, c in
                       zip(problems[i:i + bs], references[i:i + bs], candidates[i:i + bs])]
            try:
                enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=3072).to("cuda")
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            except torch.cuda.OutOfMemoryError:
                enc = gen = None
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                self.batch_size = max(1, bs // 2)
                print(f"[judge] out of memory at batch {bs} → {self.batch_size}", flush=True)
                continue
            for row in gen[:, enc["input_ids"].shape[1]:]:
                raw = self.tok.decode(row, skip_special_tokens=True)
                s = self.parse(raw)
                out.append(common.JUDGE_DEFAULT_ON_ERROR if s is None else s)   # v1 behaviour
            del enc, gen
            i += bs
        return out


JUDGE = None


def judge_correctness(prompts=None, completions=None, question=None, reference=None, completion_ids=None,
                      id=None, **kw):
    """R2 correctness: judge score in [0,1] plus R1's difficulty-aware bonus, computed from the JUDGE's verdicts
    (score ≥ 0.7 counts as correct). R1 and R2 differ only in who decides correctness."""
    texts = common._texts(completions)
    scores = JUDGE.score(question, reference, texts)
    lens = common._ntoks(texts, {"completion_ids": completion_ids})
    modes = [common.parse_policy_output(t)[0] for t in texts]
    keys = common.group_keys(len(texts), prompts, {"id": id})
    bonus = common.adaptive_bonus(keys, [s >= 0.7 for s in scores], modes, lens)
    return [s + b for s, b in zip(scores, bonus)]


# ------------------------------- balanced think/short rollouts (R1b) -------------------------
MODE_TAGS = {"think": "<think>\n", "short": "<short>\n"}


def mode_tag_ids(tok, prompt, prompt_ids):
    """Token ids of each mode tag as written after this prompt (canonical tokenisation of prompt + tag, exactly
    as TRL tokenises text prompts), and the index of the first token where <think> and <short> differ."""
    ids = {}
    for mode, tag in MODE_TAGS.items():
        full = tok(text=[prompt + tag])["input_ids"][0]
        assert list(full[:len(prompt_ids)]) == list(prompt_ids), "the tag merged into the prompt's last token"
        ids[mode] = list(full[len(prompt_ids):])
    k = next(i for i, (a, b) in enumerate(zip(ids["think"], ids["short"])) if a != b)
    return ids, k


def probe_think_prob(trainer, prefixes, think_tok, short_tok):
    """The policy's own share of <think> among the two tags, P(think) / (P(think) + P(short)), at the deciding
    token: one sampled token with the top-20 log-probs from the colocated vLLM engine. None if unavailable."""
    try:
        from vllm import SamplingParams
        outs = trainer.vllm_generation.llm.generate(
            [{"prompt_token_ids": p} for p in prefixes],
            sampling_params=SamplingParams(max_tokens=1, temperature=1.0, logprobs=20), use_tqdm=False)
        res = []
        for o, t, s in zip(outs, think_tok, short_tok):
            lp = o.outputs[0].logprobs[0]
            pt = math.exp(lp[t].logprob) if t in lp else 0.0
            ps = math.exp(lp[s].logprob) if s in lp else 0.0
            res.append(pt / (pt + ps) if pt + ps > 0 else None)
        return res
    except Exception as e:                       # a monitor must never stop training
        print("[balanced] think-probability probe failed:", repr(e)[:200], flush=True)
        return [None] * len(prefixes)


def make_balanced_rollout(tok):
    """AdaptThink-style importance-sampled exploration (Zhang et al. 2025) through TRL's rollout_func. In every
    group of G rollouts the first G/2 are forced to open <think> and the rest <short>, so both modes are always
    trained and compared on the same problem. The forced tag is part of the COMPLETION (scored and trained), and
    its mode-deciding token carries the sampling log-prob log(0.5): TRL's per-token vLLM importance correction
    (mode token_mask) then weights that token by pi_old(mode) / 0.5, the importance-sampled mode gradient. The
    other forced tag tokens carry NaN (ratio 1). Rewards are unchanged (R1's)."""
    def rollout(prompts, trainer):
        G = trainer.num_generations
        assert len(prompts) % G == 0 and all(prompts[i] == prompts[i - i % G] for i in range(len(prompts))), \
            "TRL no longer passes each prompt as G consecutive copies; balanced rollouts rely on it"
        base = [list(x) for x in trainer.processing_class(text=list(prompts))["input_ids"]]
        tags, decide, prefixes, t_tok, s_tok = {}, {}, [], [], []
        for u in range(0, len(prompts), G):
            tags[u], decide[u] = mode_tag_ids(trainer.processing_class, prompts[u], base[u])
            k = decide[u]
            prefixes.append(base[u] + tags[u]["think"][:k])
            t_tok.append(tags[u]["think"][k]); s_tok.append(tags[u]["short"][k])
        p_think = probe_think_prob(trainer, prefixes, t_tok, s_tok)
        gen_prompts, tag_of, lp_pre, forced = [], [], [], []
        for i, ids in enumerate(base):
            u = i - i % G
            mode = "think" if i % G < G // 2 else "short"
            tag = tags[u][mode]
            lp = [float("nan")] * len(tag)
            lp[decide[u]] = math.log(0.5)
            gen_prompts.append(ids + tag); tag_of.append(tag); lp_pre.append(lp); forced.append(mode)
        _, comp, logprobs, _ = trainer.vllm_generation.generate(prompts=gen_prompts, images=None, num_generations=G)
        return {"prompt_ids": base,
                "completion_ids": [t + list(c) for t, c in zip(tag_of, comp)],
                "logprobs": [pre + [l[0] for l in seq] for pre, seq in zip(lp_pre, logprobs)],
                "forced_mode": forced,
                "policy_think_prob": [p_think[i // G] for i in range(len(prompts))]}
    return rollout


# ------------------------------- callbacks --------------------------------------------------
class TimeLimit(TrainerCallback):
    """Stops (and checkpoints) after max_hours, so a long run can continue in the next session."""
    def __init__(self, hours):
        self.limit, self.t0 = hours * 3600, time.time()

    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.limit:
            print(f"[grpo] time limit reached at step {state.global_step}; saving and stopping.")
            control.should_save = True
            control.should_training_stop = True


class CollapseGuard(TrainerCallback):
    """Gate G2: record (and optionally stop) when the mode decision collapses to one mode. metric: think_rate
    (rollouts' own choice) or, for balanced runs whose rollouts are forced 50/50, policy_think_prob. The first
    collapse is written to G2_MODE_COLLAPSE.json and never overwritten."""
    def __init__(self, out, after_step, window, stop, metric="think_rate"):
        self.out, self.after, self.window, self.stop, self.metric = out, after_step, window, stop, metric
        self.hist = []

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        key = next((k for k in logs if self.metric in k and k.endswith("mean")), None)
        if key is None:
            return
        self.hist.append(float(logs[key]))
        if state.global_step % 10 == 0:
            print(f"[grpo] step {state.global_step}: {self.metric}={logs[key]:.3f} "
                  + " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in logs.items()
                             if k.startswith("rewards/") and k.endswith("/mean")), flush=True)
        if state.global_step >= self.after and len(self.hist) >= self.window:
            m = sum(self.hist[-self.window:]) / self.window
            if m < 0.05 or m > 0.95:
                f = self.out / "G2_MODE_COLLAPSE.json"
                if not f.exists():
                    msg = {"step": state.global_step, "metric": self.metric, "window_mean": m}
                    f.write_text(json.dumps(msg))
                    print("[grpo] *** G2 MODE COLLAPSE ***", msg, flush=True)
                if self.stop:
                    control.should_save = True
                    control.should_training_stop = True


def main(cfg_path):
    global JUDGE
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter" / "adapter_config.json").exists():
        print("[grpo] adapter exists, skipping:", out / "adapter")
        return
    from trl import GRPOConfig, GRPOTrainer
    import trl
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["base"])
    rows = load_jsonl(cfg["train_file"])
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "gold", "task", "question", "reference", "level", "id")}
                            for r in rows])
    base = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, cfg["init_adapter"], is_trainable=True)
    model.print_trainable_parameters()

    variant = cfg["variant"]
    if variant == "judge":
        JUDGE = Judge(cfg["judge_model"], cfg["judge_max_new_tokens"], cfg["judge_batch"], cfg.get("judge_parser", "fixed"))
        main_fn = judge_correctness
    else:
        main_fn = correctness_reward
    shaping = {"ours": [adaptive_mode_reward], "std_length": [mode_length_reward], "no_length": [],
               "judge": []}[variant]                      # R2's difficulty-aware bonus is inside judge_correctness
    monitors = [strict_accuracy, think_rate, short_rate, malformed_rate, think_correct,   # weight 0: logged only
                all_correct_group]                  # share of all-correct groups (the C2f mechanism)
    balanced = bool(cfg.get("balanced"))
    if balanced:
        monitors.append(policy_think_prob)          # rollouts are forced 50/50; this is the policy's own choice
    funcs = [main_fn, format_reward, repetition_reward] + shaping + monitors
    weights = [1] * 3 + [float(cfg.get("bonus_scale", 1.0))] * len(shaping) + [0] * len(monitors)

    gcfg = dict(
        output_dir=str(out / "checkpoints"), max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"], num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"], max_completion_length=cfg["max_completion_length"],
        temperature=1.0, top_p=1.0, repetition_penalty=1.0,
        learning_rate=cfg["lr"], lr_scheduler_type="constant_with_warmup",
        warmup_steps=max(1, int(0.05 * cfg["max_steps"])),   # warmup_ratio is ignored by TRL 1.14
        beta=cfg["beta"], reward_weights=[float(w) for w in weights],
        scale_rewards=cfg.get("scale_rewards", "group"),     # S6: "batch" (group std amplified the brevity bonus)
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=cfg["vllm_gpu_mem"],
        vllm_enable_sleep_mode=bool(cfg.get("vllm_enable_sleep_mode", False)),   # last-resort OOM fallback
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        mask_truncated_completions=False, remove_unused_columns=False,
        **({"vllm_importance_sampling_correction": True,          # balanced: per-token, so only the mode token
            "vllm_importance_sampling_mode": "token_mask"} if balanced else {}),   # gets the pi/0.5 weight
        logging_steps=1, save_steps=cfg["save_steps"], save_total_limit=2,
        report_to="tensorboard", logging_dir=str(out / "tb"), seed=cfg["seed"])
    args = GRPOConfig(**common.supported_kwargs(GRPOConfig, gcfg, required=(
        "use_vllm", "num_generations", "reward_weights", "max_completion_length", "scale_rewards")
        + (("vllm_importance_sampling_correction", "vllm_importance_sampling_mode") if balanced else ())))
    (out / "resolved_grpo_config.json").write_text(json.dumps(
        {"trl_version": trl.__version__, "variant": variant, "reward_funcs": [f.__name__ for f in funcs],
         "reward_weights": weights, "bonus_scale": float(cfg.get("bonus_scale", 1.0)), "balanced": balanced,
         "args": args.to_dict()}, indent=1, default=str))

    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=args, train_dataset=ds,
                          processing_class=tok,
                          **({"rollout_func": make_balanced_rollout(tok)} if balanced else {}),
                          callbacks=[TimeLimit(cfg["max_hours"]),
                                     CollapseGuard(out, cfg["g2_after_step"], 10, cfg["stop_on_collapse"],
                                                   "policy_think_prob" if balanced else "think_rate")])
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"))
    t0 = time.time()
    trainer.train(resume_from_checkpoint=bool(ckpts))
    finished = trainer.state.global_step >= cfg["max_steps"] or (
        cfg["stop_on_collapse"] and (out / "G2_MODE_COLLAPSE.json").exists())
    final_dir = out / ("adapter" if finished else "adapter_partial")   # partial runs stay resumable
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    (out / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "train_summary.json").write_text(json.dumps(
        {"steps": trainer.state.global_step}, indent=1))
    print(f"[grpo] saved {final_dir} (finished={finished}); rerun the cell to resume if not finished.")


if __name__ == "__main__":
    main(sys.argv[1])