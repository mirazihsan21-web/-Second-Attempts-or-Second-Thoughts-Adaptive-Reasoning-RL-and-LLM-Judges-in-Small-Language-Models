"""Packages the trained models of the study as GitHub release files.

Run it where the study's output folder is available, e.g. in Colab with Google Drive mounted (no GPU needed):
    python package_adapters.py /content/drive/MyDrive/slmreasoning /content/release_assets

For every LoRA model it writes <name>.zip (adapter_config.json, adapter_model.safetensors, model_info.json). Each
adapter applies directly to its base model (the GRPO adapters continue the SFT adapter). The DPO model was saved merged
(about 3 GB), above GitHub's 2 GB limit per release file, so it is zipped and split into parts; join them with
    cat dpo_v2_merged.zip.part* > dpo_v2_merged.zip        (Windows: copy /b dpo_v2_merged.zip.part1+dpo_v2_merged.zip.part2 dpo_v2_merged.zip)
SHA256SUMS.txt lists a checksum for every file written.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MATH, GEN = "Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
MODELS = {   # release name: (folder under runs/, base model, description)
    "sft_v2": ("sft_v2/adapter", MATH, "SFT: dual-mode LoRA SFT, think traces <= 1,500 tokens"),
    "sft_long": ("sft_long/adapter", MATH, "SFT ablation: think traces <= 3,400 tokens"),
    "grpo_ours": ("grpo_ours/adapter", MATH, "R1: GRPO, difficulty-aware mode reward, batch-std advantages (seed 42)"),
    "grpo_ours_s2": ("grpo_ours_s2/adapter", MATH, "R1, seed 43"),
    "grpo_ours_s3": ("grpo_ours_s3/adapter", MATH, "R1, seed 44"),
    "grpo_judge": ("grpo_judge/adapter", MATH, "R2: R1 with the 1.5B LLM-judge reward"),
    "grpo_stdlen": ("grpo_stdlen/adapter", MATH, "R3: R1 with the standard length bonus"),
    "grpo_balanced": ("grpo_balanced/adapter", MATH, "R1b: R1 with balanced think/short exploration (group std)"),
    "grpo_ours_gstd": ("grpo_ours_gstd/adapter", MATH, "R1 with group-std advantages (collapsed late in training)"),
    "grpo_ours_gstd_step60": ("grpo_ours_gstd/adapter_g2stop_step60", MATH,
                              "R1 with group-std advantages, step-60 adapter (before the collapse)"),
    "grpo_stdlen_gstd": ("grpo_stdlen_gstd/adapter", MATH, "R3 with group-std advantages"),
    "grpo_nobonus_gstd": ("grpo_nobonus_gstd/adapter", MATH, "Control C1: group std, no brevity bonus"),
    "grpo_ours_nostd": ("grpo_ours_nostd/adapter", MATH, "Control C2: R1's bonus, no std scaling"),
    "grpo_ours_gstd_b01": ("grpo_ours_gstd_b01/adapter", MATH, "Control C3: group std, R1's bonus x 0.1"),
    "m2_sft": ("m2_sft/adapter", GEN, "Second base model: SFT"),
    "m2_grpo": ("m2_grpo/adapter", GEN, "Second base model: R1 (batch std)"),
    "m2_grpo_gstd": ("m2_grpo_gstd/adapter", GEN, "Second base model: R1 with group-std advantages"),
}
PART = 1_900_000_000          # below GitHub's 2 GB limit per release file


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
written, missing = [], []
for name, (folder, base, desc) in MODELS.items():
    d = ROOT / "runs" / folder
    files = [d / "adapter_config.json", d / "adapter_model.safetensors"]
    if not all(f.exists() for f in files):
        missing.append(f"{name} ({d})")
        continue
    cfg = json.loads(files[0].read_text())
    info = dict(name=name, description=desc, base_model=base, lora_r=cfg.get("r"), lora_alpha=cfg.get("lora_alpha"),
                target_modules=cfg.get("target_modules"),
                prompt="common.build_prompt(tokenizer, question, 'policy') from src/common.py")
    z = OUT / f"{name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f"{name}/{f.name}")
        zf.writestr(f"{name}/model_info.json", json.dumps(info, indent=1))
    written.append(z)
    print(f"{name}: {z.stat().st_size / 1e6:.0f} MB")

merged = ROOT / "runs" / "dpo_v2" / "merged"
if (merged / "config.json").exists():
    z = OUT / "dpo_v2_merged.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:      # safetensors barely compress
        for f in sorted(merged.iterdir()):
            if f.is_file():
                zf.write(f, f"dpo_v2_merged/{f.name}")
        zf.writestr("dpo_v2_merged/model_info.json", json.dumps(dict(
            name="dpo_v2", base_model=MATH, description="DPO from the SFT model, saved merged (a full model)",
            prompt="common.build_prompt(tokenizer, question, 'policy') from src/common.py"), indent=1))
    if z.stat().st_size > PART:
        with z.open("rb") as f:
            i = 1
            while chunk := f.read(PART):
                part = OUT / f"{z.name}.part{i}"
                part.write_bytes(chunk)
                written.append(part)
                i += 1
        whole = sha256(z)
        z.unlink()
        (OUT / "dpo_v2_merged.zip.sha256").write_text(f"{whole}  dpo_v2_merged.zip\n")
        print(f"dpo_v2: split into {i - 1} parts (checksum of the joined zip in dpo_v2_merged.zip.sha256)")
    else:
        written.append(z)
else:
    missing.append(f"dpo_v2 ({merged})")

(OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in written))
print(f"\n{len(written)} files in {OUT}; checksums in SHA256SUMS.txt")
if missing:
    print("NOT FOUND (check the folder):", *missing, sep="\n  ")
