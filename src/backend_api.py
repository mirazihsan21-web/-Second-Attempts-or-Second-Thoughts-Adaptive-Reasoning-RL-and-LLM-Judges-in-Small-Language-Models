# backend_api.py: single-pass adaptive inference server for the v2 GRPO adapter.
# The model chooses <think> or <short> itself in ONE generation (no multi-call workflow,
# no second model rewriting the answer). Run:  MODEL_DIR=./adapter uvicorn backend_api:app
import os
from contextlib import asynccontextmanager
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Math-1.5B-Instruct")
ADAPTER = os.getenv("MODEL_DIR", "./adapter")
STATE = {}
LOCK = Lock()


@asynccontextmanager
async def lifespan(app):
    tok = AutoTokenizer.from_pretrained(BASE)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
    if torch.cuda.is_available():
        model = model.cuda()
    STATE.update(tok=tok, model=model)
    yield
    STATE.clear()


app = FastAPI(title="Adaptive reasoning SLM", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    max_new_tokens: int = Field(default=2048, ge=16, le=3584)
    force_mode: str | None = Field(default=None, pattern="^(think|short)$")


@app.post("/solve")
def solve(q: Query):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    tok, model = STATE["tok"], STATE["model"]
    prefill = f"<{q.force_mode}>\n" if q.force_mode else ""
    prompt = common.build_prompt(tok, q.question, "policy", prefill)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with LOCK, torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=q.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    text = prefill + tok.decode(gen, skip_special_tokens=True)
    mode, answer, status = common.extract_final(text, "policy")
    _, _, think = common.parse_policy_output(text)
    return {"mode": mode, "final_answer": answer, "extraction": status, "reasoning": think,
            "raw": text, "generated_tokens": int(gen.shape[0]),
            "truncated": int(gen.shape[0]) >= q.max_new_tokens}


@app.get("/health")
def health():
    return {"loaded": "model" in STATE, "base": BASE, "adapter": ADAPTER}