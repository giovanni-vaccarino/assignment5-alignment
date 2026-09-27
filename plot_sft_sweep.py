"""
Plot the SFT sweeps from outputs/sft/<run>/ (metrics.jsonl + eval/step_*.jsonl).

    uv run --no-sync python plot_sft_sweep.py --full-run full_lr2e-5

For each sweep (Phase 1 lr/bs, dataset size) writes:
    outputs/sft/<sweep>.png          eval accuracy vs step (the assignment deliverable)
    outputs/sft/<sweep>_metrics.png  accuracy, format rate, response length (correct / incorrect),
                                     token entropy, train loss per token, grad norm
Response length is recomputed over the full eval set from eval/step_*.jsonl (the metrics.jsonl value
comes from only n_log_generations=16 samples); token entropy is only available for those 16.
Runs that don't exist yet are skipped.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

PHASE1_RUNS = ["full_lr1e-5", "full_lr2e-5", "full_lr5e-5", "full_lr2e-5_bs64"]
SIZE_RUNS = ["n128", "n256", "n512", "n1024"]


def load_metrics(run_dir: Path, split: str) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return sorted((r for r in rows if r["split"] == split), key=lambda r: r["step"])


def eval_lengths(run_dir: Path, tokenizer) -> list[dict]:
    """Mean response length in tokens over the full eval set, per eval step, split by correctness."""
    rows = []
    for path in sorted((run_dir / "eval").glob("step_*.jsonl")):
        results = [json.loads(line) for line in path.open() if line.strip()]
        if not results:
            continue
        lengths = [len(ids) for ids in tokenizer([r["response"] for r in results])["input_ids"]]
        correct = [n for n, r in zip(lengths, results) if r["rewards"]["answer_reward"] == 1.0]
        wrong = [n for n, r in zip(lengths, results) if r["rewards"]["answer_reward"] != 1.0]
        rows.append({
            "step": int(path.stem.split("_")[1]),
            "all": sum(lengths) / len(lengths),
            "correct": sum(correct) / len(correct) if correct else float("nan"),
            "incorrect": sum(wrong) / len(wrong) if wrong else float("nan"),
        })
    return rows


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average (train metrics are noisy per step)."""
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def plot_accuracy(runs: dict[str, list[dict]], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in runs.items():
        acc = [r["accuracy"] for r in rows]
        ax.plot([r["step"] for r in rows], acc, marker="o", markersize=3, label=f"{label} (final {acc[-1]:.3f})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("eval accuracy (GSM8K test)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_metrics(run_dirs: dict[str, Path], title: str, out_path: Path, tokenizer, smooth_window: int):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    (ax_acc, ax_fmt, ax_len), (ax_ent, ax_loss, ax_gn) = axes
    for i, (label, run_dir) in enumerate(run_dirs.items()):
        color = f"C{i}"
        ev = load_metrics(run_dir, "eval")
        tr = load_metrics(run_dir, "train")
        ev_steps = [r["step"] for r in ev]
        ax_acc.plot(ev_steps, [r["accuracy"] for r in ev], color=color, marker="o", markersize=3, label=label)
        ax_fmt.plot(ev_steps, [r["format_rate"] for r in ev], color=color, marker="o", markersize=3, label=label)
        ax_ent.plot(ev_steps, [r["mean_token_entropy"] for r in ev], color=color, marker="o", markersize=3,
                    label=label)
        if lengths := eval_lengths(run_dir, tokenizer):
            steps = [r["step"] for r in lengths]
            ax_len.plot(steps, [r["correct"] for r in lengths], color=color, marker="o", markersize=3,
                        label=f"{label} correct")
            ax_len.plot(steps, [r["incorrect"] for r in lengths], color=color, marker="x", markersize=4,
                        linestyle="--", label=f"{label} incorrect")
        if tr:
            tr_steps = [r["step"] for r in tr]
            ax_loss.plot(tr_steps, smooth([r["loss_per_token"] for r in tr], smooth_window), color=color,
                         label=label)
            ax_gn.plot(tr_steps, smooth([r["grad_norm"] for r in tr], smooth_window), color=color, label=label)

    ax_acc.set_title("eval accuracy")
    ax_fmt.set_title("eval format rate")
    ax_len.set_title("eval response length (tokens, full eval set)")
    ax_ent.set_title("mean response token entropy (16 eval samples)")
    ax_loss.set_title(f"train loss per response token (moving avg {smooth_window})")
    ax_gn.set_title(f"train grad norm before clipping (moving avg {smooth_window})")
    for ax in axes.flat:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(
    output_dir: str = "outputs/sft",
    full_run: str = "full_lr2e-5",
    tokenizer_name: str = "Qwen/Qwen2.5-Math-1.5B",
    smooth_window: int = 20,
):
    """full_run: the Phase 1 run used as the full-dataset point of the dataset-size sweep (BEST)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    out = Path(output_dir)

    phase1 = {r: out / r for r in PHASE1_RUNS if load_metrics(out / r, "eval")}
    sizes = {r: out / r for r in SIZE_RUNS if load_metrics(out / r, "eval")}
    if load_metrics(out / full_run, "eval"):
        sizes[f"full ({full_run})"] = out / full_run

    for name, runs, title in [
        ("phase1_lr_sweep", phase1, "Phase 1: lr / batch-size sweep (full train set)"),
        ("dataset_size_sweep", sizes, "Dataset-size sweep"),
    ]:
        if not runs:
            continue
        plot_accuracy({k: load_metrics(v, "eval") for k, v in runs.items()}, title, out / f"{name}.png")
        plot_metrics(runs, title, out / f"{name}_metrics.png", tokenizer, smooth_window)

    print(f"\n{'run':<28} {'final_acc':>9} {'best_acc':>9} {'final_fmt':>9}")
    for label, run_dir in {**phase1, **sizes}.items():
        rows = load_metrics(run_dir, "eval")
        print(f"{label:<28} {rows[-1]['accuracy']:>9.3f} {max(r['accuracy'] for r in rows):>9.3f} "
              f"{rows[-1]['format_rate']:>9.3f}")


if __name__ == "__main__":
    typer.run(main)
