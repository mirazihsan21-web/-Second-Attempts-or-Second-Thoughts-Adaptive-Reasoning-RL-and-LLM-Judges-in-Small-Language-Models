# Second Attempts or Second Thoughts?

Code, trained models and outputs for the paper
**"Second Attempts or Second Thoughts? Adaptive Reasoning, Reinforcement Learning, and LLM Judges in Small Language
Models"** by Md. Miraz Ihsan Chowdhury, Abdus Samad Sunny and Mohammad Ashrafuzzaman Khan (Department of Electrical
and Computer Engineering, North South University, Dhaka, Bangladesh).

We teach a 1.5B instruct model (Qwen2.5-Math-1.5B-Instruct, replicated on Qwen2.5-1.5B-Instruct) a dual-mode format,
`<think>` for long reasoning and `<short>` for its native short reasoning, with LoRA SFT, and then train the choice of
mode with GRPO. We compare thinking longer (second thoughts) with sampling several answers and voting (second attempts),
and test a 1.5B LLM judge as the RL reward. All accuracies come from a strict rule-based scorer.

## Main findings
- **Thinking does not pay off at 1.5B, and GRPO learns this.** On the same problems, a forced think answer is never
  significantly more accurate than a forced short answer, for either base model. Every GRPO run stops choosing the think
  mode early in training; the trained model is as accurate as its base model.
- **Group-normalized advantages turn a small brevity bonus into a collapse of the reasoning.** With TRL's default
  `scale_rewards="group"`, completions shrank by three quarters late in training and accuracy fell by 20 to 25 points.
  Three controls show that both the normalization and the bonus are needed and that the bonus's size does not matter;
  normalizing by the batch standard deviation (`scale_rewards="batch"`) prevents the collapse.
- **A small LLM judge is a usable reward once its output is read correctly.** A lenient parser makes the 1.5B judge
  accept 91% of wrong answers; reading its actual verdict gives 15%. Trained with this judge instead of the answer
  checker, the policy is as accurate as with the checker.
- **Second attempts beat second thoughts, but not against a trained adaptive thinker.** At matched token counts, a
  vote over short answers beats the model's own thinking by 9.6 points on MATH-500 and beats
  DeepSeek-R1-Distill-Qwen-1.5B, but not AdaptThink on MATH-500.

## Repository contents
| Path | What |
|---|---|
| `notebook/slm_adaptive_reasoning.ipynb` | The whole study in one Colab notebook: data, baselines, judge study, SFT, GRPO variants, DPO, evaluation, votes and MPV, the second base model, all tables and figures |
| `src/` | The modules the notebook writes and runs: scorer, prompts and rewards (`common.py`), vLLM generation (`gen_worker.py`), GRPO (`train_grpo.py`), SFT as a separate process for the second base model (`train_sft.py`), a small inference server (`backend_api.py`). The main SFT and DPO run inside the notebook |
| `results/eval/preds/` | Every model's answers on GSM8K, MATH-500 and AIME 2026, with strict-scorer verdicts (gzipped JSON lines), plus votes, MPV and judge outputs |
| `results/paper_tables/`, `results/paper_figures/` | The notebook's result tables (CSV/LaTeX) and figures |
| `results/runs/` | Training logs and resolved configurations of every run (no weights) |
| `results/data/` | The 3,000 verified SFT pairs (train/eval split), the RL and DPO prompts, data statistics |
| `paper_scripts/` | Scripts that build the paper's tables and figures from `results/` |
| `tools/package_adapters.py` | Packages the trained adapters from the study's output folder as release files |

Released logs and predictions omit timing fields.

## Trained models
The adapters are attached to the GitHub release (one zip per model). Each LoRA adapter loads directly on its base model.

| File | Base model | Model |
|---|---|---|
| `sft_v2.zip` | Qwen2.5-Math-1.5B-Instruct | SFT (think traces up to 1,500 tokens) |
| `sft_long.zip` | Qwen2.5-Math-1.5B-Instruct | SFT ablation (think traces up to 3,400 tokens) |
| `grpo_ours.zip`, `grpo_ours_s2.zip`, `grpo_ours_s3.zip` | Qwen2.5-Math-1.5B-Instruct | R1: GRPO with the difficulty-aware mode reward, batch-std advantages; seeds 42, 43, 44 |
| `grpo_judge.zip` | Qwen2.5-Math-1.5B-Instruct | R2: R1 with the 1.5B LLM-judge reward |
| `grpo_stdlen.zip` | Qwen2.5-Math-1.5B-Instruct | R3: R1 with a standard length bonus |
| `grpo_balanced.zip` | Qwen2.5-Math-1.5B-Instruct | R1b: balanced think/short exploration |
| `grpo_ours_gstd.zip`, `grpo_ours_gstd_step60.zip`, `grpo_stdlen_gstd.zip` | Qwen2.5-Math-1.5B-Instruct | R1 and R3 with group-std advantages (collapsed); R1's step-60 adapter before the collapse |
| `grpo_nobonus_gstd.zip`, `grpo_ours_nostd.zip`, `grpo_ours_gstd_b01.zip` | Qwen2.5-Math-1.5B-Instruct | Controls C1, C2, C3 |
| `dpo_v2_merged.zip.part*` | Qwen2.5-Math-1.5B-Instruct | DPO (a merged full model; join the parts first, see below) |
| `m2_sft.zip`, `m2_grpo.zip`, `m2_grpo_gstd.zip` | Qwen2.5-1.5B-Instruct | Second base model: SFT, R1, R1 with group std |

`SHA256SUMS.txt` lists the checksums. Join the DPO parts with `cat dpo_v2_merged.zip.part* > dpo_v2_merged.zip`
(Windows: `copy /b dpo_v2_merged.zip.part1+dpo_v2_merged.zip.part2 dpo_v2_merged.zip`).

## Quick start
```bash
pip install torch transformers peft math-verify
```
```python
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, "src")
import common                                   # the prompt builder used for SFT, GRPO and evaluation

base = "Qwen/Qwen2.5-Math-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "grpo_ours")       # the unzipped release file

question = "A bakery sells 12 muffins per tray. How many muffins are on 7 trays?"
prefill = ""                                    # the model chooses its mode; "<think>\n" or "<short>\n" forces one
prompt = common.build_prompt(tok, question, "policy", prefill=prefill)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=3584, do_sample=False)
response = prefill + tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
mode, answer, status = common.extract_final(response)       # e.g. ('short', '84', 'boxed')
print(response, "\n->", mode, answer)
```
Scoring against a gold answer: `common.score_text(response, gold, task)` with `task` in `gsm8k`, `math500`, `aime2026`.

## Reproducing the study
Open `notebook/slm_adaptive_reasoning.ipynb` in Google Colab with an A100 GPU and follow the instructions in its first
cell. Cell 5 sets `SCALE` (`"pilot"`: a small wiring test of every stage; `"full"`: the paper study) and `SESSION`. The
study ran as sessions S1 to S11 (plus S2b), one "Run all" each; the notebook applies the settings that changed between
sessions (the first SFT's longer traces, and group-std advantages in S3 to S5). All outputs are cached on Google Drive,
and finished stages skip themselves. GPU decoding is not bit-exact across hardware and library versions, so a rerun can
differ slightly from the released outputs. Library versions: PyTorch 2.11, Transformers 5.18, TRL 1.15, PEFT 0.21,
vLLM 0.26, math-verify 0.9. The study's measured energy use was 8.76 kWh (3.48 kg CO2eq, CodeCarbon).

## Rebuilding the paper's tables and figures
```bash
pip install pandas matplotlib math-verify
python paper_scripts/make_tables.py results paper_tables_out
python paper_scripts/make_figures.py results src paper_figures_out
```

## Data and licenses
- Code: Apache License 2.0 (`LICENSE`).
- Adapters: derived from Qwen2.5-Math-1.5B-Instruct and Qwen2.5-1.5B-Instruct (both Apache 2.0).
- SFT pairs: selected from [Vinnnf/Hybrid-OpenThoughts2-1M-1.5B](https://huggingface.co/datasets/Vinnnf/Hybrid-OpenThoughts2-1M-1.5B)
  (Apache 2.0). RL and DPO prompts: the MATH and GSM8K training sets (MIT).
- Benchmarks are not redistributed: GSM8K (MIT), MATH-500, and AIME 2026 from
  [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) (CC BY-NC-SA 4.0). The notebook downloads
  them at pinned revisions; the released predictions contain model outputs and verdicts, not the problems.

## Citation
If you use this code, the models or the outputs, please cite the paper (publication details will be added here):
```bibtex
@misc{chowdhury2026second,
  title  = {Second Attempts or Second Thoughts? Adaptive Reasoning, Reinforcement Learning, and LLM Judges in
            Small Language Models},
  author = {Chowdhury, Md. Miraz Ihsan and Sunny, Abdus Samad and Khan, Mohammad Ashrafuzzaman},
  year   = {2026}
}
```
