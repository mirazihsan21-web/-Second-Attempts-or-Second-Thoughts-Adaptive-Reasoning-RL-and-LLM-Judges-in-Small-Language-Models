# train_sft.py: Stage A. LoRA SFT on verified <think>/<short> examples (both modes kept).
# Usage: python train_sft.py config.json
# Reuses the v1 recipe that was correct (LoRA r16/a32 on all projections, prompt-masked labels,
# DataCollatorForSeq2Seq) and fixes the v1 bug that dropped every <think> example.
import json, math, os, sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    if (out / "adapter" / "adapter_config.json").exists():
        print("[sft] adapter exists, skipping:", out / "adapter")
        return
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    end_ids = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def encode(ex):
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(ex["target"], add_special_tokens=False)["input_ids"] + end_ids
        return {"input_ids": p + t, "attention_mask": [1] * (len(p) + len(t)),
                "labels": [-100] * len(p) + t}

    train = Dataset.from_list(load_jsonl(cfg["train_file"])).map(encode, remove_columns=["prompt", "target", "mode", "id"])
    evald = Dataset.from_list(load_jsonl(cfg["eval_file"])).map(encode, remove_columns=["prompt", "target", "mode", "id"])
    for d in (train, evald):
        too_long = sum(len(x) > cfg["max_len"] for x in d["input_ids"])
        assert too_long == 0, f"{too_long} examples exceed max_len; P1 filtering must remove them."
    print(f"[sft] train={len(train)} eval={len(evald)}")

    # The same recipe as the notebook's inline SFT (cells 9–15), used for the second base model (S10). The OOM retry
    # (run_adaptive) halves the micro-batch and doubles grad-accum, so the effective batch and the step count stay fixed.
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16, attn_implementation=attn)
    print(f"[sft] loaded {cfg['base']} (attn={attn}), pad={tok.pad_token!r}, eos={tok.eos_token!r}")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    max_steps = cfg.get("max_steps", -1)
    total = max_steps if max_steps > 0 else math.ceil(len(train) / (cfg["batch"] * cfg["grad_accum"])) * cfg["epochs"]
    args = TrainingArguments(**common.supported_kwargs(TrainingArguments, dict(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg["epochs"], max_steps=max_steps,
        per_device_train_batch_size=cfg["batch"], per_device_eval_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total)),   # warmup_ratio is ignored by this stack
        max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch_fused",
        group_by_length=True, eval_strategy="steps", save_strategy="steps",
        eval_steps=cfg["eval_steps"], save_steps=cfg["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=5, report_to="tensorboard", logging_dir=str(out / "tb"),
        dataloader_num_workers=2, seed=cfg["seed"]),
        required=("learning_rate", "bf16", "eval_strategy")))
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evald,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(ckpts))
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "train_summary.json").write_text(json.dumps({"base": cfg["base"], "steps": trainer.state.global_step,
                                                         "train": len(train), "eval": len(evald)}, indent=1))
    print("[sft] saved", out / "adapter")


if __name__ == "__main__":
    main(sys.argv[1])