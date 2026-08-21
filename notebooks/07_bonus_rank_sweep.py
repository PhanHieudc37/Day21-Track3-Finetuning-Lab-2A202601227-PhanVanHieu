# %% [markdown]
# # Bonus B4 — Quét rank có kiểm soát
#
# Giữ cố định toàn bộ cấu hình `correct` và chỉ đổi rank `r ∈ {8, 16, 64}`.
# Run r=16 được tái sử dụng từ NB3 vì nó đã có đúng placement, LR, dữ liệu và 30 step;
# việc huấn luyện lại cùng một cấu hình không tạo thêm thông tin nhưng tốn một lượt GPU.

# %%
import dataclasses
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import SPECS, get_tier, training_epochs

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
RANKS = (8, 16, 64)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
if os.environ.get("EVAL_LIMIT", "").strip():
    raise SystemExit("Bonus B4 phải chấm trên toàn bộ eval; hãy bỏ EVAL_LIMIT rồi chạy lại.")


def read_previous_rows():
    rows = report.read_rows("rank_sweep.csv", results_dir=ROOT / "results")
    return {int(r["r"]): r for r in rows if r.get("r")}


def reuse_core_r16():
    core_runs = {
        r["run"]: r for r in report.read_rows("runs.csv", results_dir=ROOT / "results")
        if r.get("run")
    }
    autopsy = {
        r["run"]: r
        for r in json.loads((ROOT / "results" / "autopsy.json").read_text(encoding="utf-8"))
    }
    core = core_runs.get("correct")
    score = autopsy.get("correct")
    if not core or not score:
        raise SystemExit("Thiếu run correct/autopsy của pipeline lõi; hãy khôi phục results/ trước.")
    if (core.get("placement") != "text-linear" or int(core.get("r", 0)) != 16
            or float(core.get("learning_rate", 0)) != float(SPECS["correct"].lr)):
        raise SystemExit("Run correct không khớp cấu hình điều khiển của B4.")
    return {
        "run": "rank_r16",
        "r": 16,
        "placement": core["placement"],
        "learning_rate": float(core["learning_rate"]),
        "max_steps": int(core["max_steps"]),
        "trainable_params": int(core["trainable_params"]),
        "final_loss": float(core["final_loss"]),
        "target": float(score["target"]),
        "format": float(score["format"]),
        "train_seconds": float(core["train_seconds"]),
        "peak_vram_gb": float(core["peak_vram_gb"]),
        "source": "reused_core_correct",
    }


train_ds = None


def train_and_score(rank):
    global train_ds
    key = f"rank_r{rank}"
    spec = dataclasses.replace(
        SPECS["correct"], key=key, r=rank, alpha=2 * rank,
        label=f"text-linear · r={rank} · LR 10x · 16-bit",
        teaches="B4 controlled rank sweep",
    )
    model, tok = generate.load_base(TIER)
    if train_ds is None:
        train_ds = Dataset.from_list(data.to_training_dataset(
            tok, train_rows, max_length=TIER.max_length, mask_mode="assistant-only"
        ))

    targets = modeling.resolve_target_modules(model, "text-linear")
    trainable = modeling.count_lora_params(model, targets, rank)
    max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())
    want = train.sft_config_kwargs(
        TIER, spec, str(ROOT / "adapters" / key), max_steps=max_steps
    )
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]"
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=LoraConfig(**lora_kwargs),
    )
    train.align_trainable_precision(trainer.model)
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started

    adapter_dir = ROOT / "adapters" / key
    trainer.model.save_pretrained(adapter_dir)
    trainer.model.eval()
    predictions, latency = generate.generate_batch(
        trainer.model, tok, [r["input"] for r in target],
        system=generate.NAIVE_PROMPT, label=f"{key}/target",
    )
    target_score = sum(
        ev.triage_field_accuracy(pred, row["label"])
        for pred, row in zip(predictions, target)
    ) / len(target)
    format_score = sum(
        ev.has_required_keys(pred, ev.TRIAGE_KEYS) for pred in predictions
    ) / len(predictions)

    row = {
        "run": key,
        "r": rank,
        "placement": "text-linear",
        "learning_rate": spec.lr,
        "max_steps": max_steps,
        "trainable_params": trainable,
        "final_loss": round(result.training_loss, 4),
        "target": round(target_score, 4),
        "format": round(format_score, 4),
        "latency_ms": round(latency, 1),
        "train_seconds": round(elapsed, 1),
        "peak_vram_gb": generate.peak_vram_gb(),
        "source": "trained_in_bonus_b4",
    }
    report.append_row(row, "rank_sweep.csv", results_dir=ROOT / "results")
    del trainer, model
    generate.free_memory()
    return row


# %% [markdown]
# ## Chạy có thể resume

# %%
previous = read_previous_rows()
rows = {16: reuse_core_r16(), **previous}
for rank in (8, 64):
    adapter_ok = (ROOT / "adapters" / f"rank_r{rank}" / "adapter_model.safetensors").exists()
    if rank in rows and adapter_ok:
        print(f"skip r={rank}: đã có adapter và kết quả")
        continue
    print("=" * 70)
    print(f"B4 train r={rank}; placement=text-linear; LR={SPECS['correct'].lr}; steps=30")
    rows[rank] = train_and_score(rank)

ordered = [rows[r] for r in RANKS if r in rows]
if len(ordered) != len(RANKS):
    raise SystemExit("B4 chưa đủ ba rank 8, 16, 64.")

steps = {int(r["max_steps"]) for r in ordered}
placements = {r["placement"] for r in ordered}
lrs = {float(r["learning_rate"]) for r in ordered}
assert len(steps) == len(placements) == len(lrs) == 1

targets = [float(r["target"]) for r in ordered]
rank_effect = max(targets) - min(targets)
core_autopsy = {
    r["run"]: r
    for r in json.loads((ROOT / "results" / "autopsy.json").read_text(encoding="utf-8"))
}
placement_effect = abs(float(core_autopsy["correct"]["target"])
                       - float(core_autopsy["attn_only"]["target"]))
lr_effect = abs(float(core_autopsy["correct"]["target"])
                - float(core_autopsy["wrong_lr"]["target"]))

payload = {
    "controlled_variables": {
        "placement": ordered[0]["placement"],
        "learning_rate": float(ordered[0]["learning_rate"]),
        "max_steps": int(ordered[0]["max_steps"]),
        "ranks": list(RANKS),
        "eval_n": len(target),
    },
    "results": ordered,
    "effect_size_target": {
        "rank_sweep_range": round(rank_effect, 4),
        "placement_correct_vs_attn_only": round(placement_effect, 4),
        "learning_rate_correct_vs_wrong_lr": round(lr_effect, 4),
    },
}
report.write_json(payload, "rank_sweep.json", results_dir=ROOT / "results")
print(report.markdown_table(ordered, ["run", "r", "target", "format", "final_loss",
                                      "trainable_params", "max_steps"]))
print("effect sizes:", payload["effect_size_target"])
print("đã ghi results/rank_sweep.json và results/rank_sweep.csv")

# %% [markdown]
# ## Câu hỏi cần trả lời trong report
#
# So biên độ target của rank với đối chứng vị trí và LR. Rank chỉ là đòn bẩy khi dữ
# liệu chứa đủ thông tin nhưng capacity của adapter thấp đang là nút thắt; nếu tăng rank
# không tạo chênh lệch target đáng kể, thêm capacity chỉ làm adapter lớn hơn.
