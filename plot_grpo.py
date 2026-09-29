"""
Plot GRPO runs from outputs/grpo/<run>/ (metrics.jsonl, eval/step_*.jsonl, generations.jsonl).

    HF_HOME=/projects/bhag/gvaccarino/hf_cache uv run --no-sync python plot_grpo.py grpo_base [other runs...] \
        --name base

Writes (one line per run, so ablations overlay on the same axes):
    outputs/grpo/<name>_eval_reward.png   headline: eval answer reward vs GRPO step, with SFT / zero-shot references
    outputs/grpo/<name>_metrics.png       3x4 grid: rewards (eval + train rollouts), zero-std groups, response
                                          lengths, loss, grad norm, entropy, clip fraction, wall-clock and timing
and per run:
    outputs/grpo/<run>/rollouts_over_time.md   a few eval samples at the first / middle / last eval step

x-axis is the GRPO step (one rollout batch) everywhere. Train metrics are logged per optimizer step; they are
averaged per GRPO step so on-policy and off-policy runs share an axis.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

# Reference points on the same eval (GSM8K test, r1_zero prompt, T=1), from SFT_EXPERIMENTS.md
REFERENCES = {"zero-shot base (2.5%)": 0.025, "SFT full data (62.7%)": 0.627}


def load_split(run_dir: Path, split: str) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return sorted((r for r in rows if r["split"] == split), key=lambda r: r["step"])


def train_per_grpo_step(run_dir: Path) -> list[dict]:
    """Average the per-optimizer-step train metrics within each GRPO step."""
    groups = defaultdict(list)
    for r in load_split(run_dir, "train"):
        groups[int(r["grpo_step"])].append(r)
    out = []
    for step in sorted(groups):
        rows = groups[step]
        avg = {"step": step}
        for key in ("loss", "grad_norm", "token_entropy", "clip_fraction", "step_time_s"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                avg[key] = sum(vals) / len(vals)
        avg["train_time_s"] = sum(r["step_time_s"] for r in rows)
        out.append(avg)
    return out


def eval_lengths(run_dir: Path, tokenizer) -> list[dict]:
    """Mean response length (tokens) over the full eval set per eval step, split by correctness."""
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
            "correct": sum(correct) / len(correct) if correct else float("nan"),
            "incorrect": sum(wrong) / len(wrong) if wrong else float("nan"),
        })
    return rows


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average (per-step rollout metrics are noisy: 32 questions per step)."""
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def xy(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])


def plot_eval_reward(runs: dict[str, Path], out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, run_dir) in enumerate(runs.items()):
        ev = load_split(run_dir, "eval")
        x, y = xy(ev, "answer_reward")
        if y:
            ax.plot(x, y, color=f"C{i}", marker="o", markersize=3, label=f"{label} (last {y[-1]:.3f}, best {max(y):.3f})")
    for j, (name, val) in enumerate(REFERENCES.items()):
        ax.axhline(val, color="gray", linestyle=[":", "--"][j % 2], linewidth=1, label=name)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("eval answer reward (GSM8K test accuracy)")
    ax.set_title("GRPO: eval answer reward")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_metrics(runs: dict[str, Path], out_path: Path, tokenizer, window: int):
    fig, axes = plt.subplots(3, 4, figsize=(21, 12.5))
    ax = {name: a for name, a in zip(
        ["eval_ans", "eval_fmt", "roll_rew", "roll_fmt",
         "zero_std", "roll_len", "eval_len", "entropy",
         "loss", "grad_norm", "wall", "timing"], axes.flat)}
    has_clip = False
    for i, (label, run_dir) in enumerate(runs.items()):
        c = f"C{i}"
        ev = load_split(run_dir, "eval")
        ro = load_split(run_dir, "rollout")
        tr = train_per_grpo_step(run_dir)
        kw = dict(color=c, marker="o", markersize=3)

        ax["eval_ans"].plot(*xy(ev, "answer_reward"), label=label, **kw)
        ax["eval_fmt"].plot(*xy(ev, "format_reward"), label=label, **kw)

        # Train-time rewards on the rollouts (32 questions x 8 samples per step): raw faint + moving average bold
        for key, name in [("reward", "roll_rew"), ("format_reward", "roll_fmt")]:
            x, y = xy(ro, key)
            if y:
                ax[name].plot(x, y, color=c, alpha=0.2, linewidth=0.8)
                ax[name].plot(x, smooth(y, window), color=c, label=label)
        x, y = xy(ro, "frac_zero_std_groups")
        if y:
            ax["zero_std"].plot(x, y, color=c, alpha=0.2, linewidth=0.8)
            ax["zero_std"].plot(x, smooth(y, window), color=c, label=label)

        for key, style, suffix in [("response_length_correct", "-", "correct"),
                                   ("response_length_incorrect", "--", "incorrect")]:
            x, y = xy(ro, key)
            if y:
                ax["roll_len"].plot(x, smooth(y, window), color=c, linestyle=style, label=f"{label} {suffix}")
        if lengths := eval_lengths(run_dir, tokenizer):
            steps = [r["step"] for r in lengths]
            ax["eval_len"].plot(steps, [r["correct"] for r in lengths], label=f"{label} correct", **kw)
            ax["eval_len"].plot(steps, [r["incorrect"] for r in lengths], color=c, marker="x", markersize=4,
                                linestyle="--", label=f"{label} incorrect")

        x, y = xy(tr, "token_entropy")
        if y:
            ax["entropy"].plot(x, smooth(y, window), color=c, label=f"{label} train rollouts")
        ax["entropy"].plot(*xy(ev, "mean_token_entropy"), color=c, marker="o", markersize=3, linestyle=":",
                           label=f"{label} eval (16 samples)")

        for key, name in [("loss", "loss"), ("grad_norm", "grad_norm")]:
            x, y = xy(tr, key)
            if y:
                ax[name].plot(x, y, color=c, alpha=0.2, linewidth=0.8)
                ax[name].plot(x, smooth(y, window), color=c, label=label)
        x, y = xy(tr, "clip_fraction")
        if y:
            has_clip = True
            ax["grad_norm"].plot(x, smooth(y, window), color=c, linestyle=":", label=f"{label} clip fraction")

        if ev and "wall_clock_s" in ev[0]:
            ax["wall"].plot([r["wall_clock_s"] / 60 for r in ev], [r["answer_reward"] for r in ev], label=label, **kw)
        x, y = xy(ro, "rollout_time_s")
        if y:
            ax["timing"].plot(x, smooth(y, window), color=c, label=f"{label} rollout")
        x, y = xy(tr, "train_time_s")
        if y:
            ax["timing"].plot(x, smooth(y, window), color=c, linestyle="--", label=f"{label} train")

    titles = {
        "eval_ans": "eval answer reward (accuracy, full test set)",
        "eval_fmt": "eval format reward",
        "roll_rew": f"train rollout reward (moving avg {window})",
        "roll_fmt": f"train rollout format reward (moving avg {window})",
        "zero_std": f"fraction of groups with zero reward std\n(all 8 right or all 8 wrong: no gradient) (avg {window})",
        "roll_len": f"train rollout response length, tokens (avg {window})",
        "eval_len": "eval response length, tokens (full test set)",
        "entropy": "mean response token entropy",
        "loss": f"policy-gradient loss (avg {window})",
        "grad_norm": f"grad norm before clipping (avg {window})" + (" / clip fraction" if has_clip else ""),
        "wall": "eval answer reward vs wall-clock (min)",
        "timing": f"seconds per GRPO step (avg {window})",
    }
    for name, a in ax.items():
        a.set_title(titles[name], fontsize=10)
        a.set_xlabel("wall-clock (min)" if name == "wall" else "GRPO step")
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    for name in ("eval_ans", "roll_rew"):
        for j, (ref, val) in enumerate(REFERENCES.items()):
            ax[name].axhline(val, color="gray", linestyle=[":", "--"][j % 2], linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_rollouts_over_time(run_dir: Path, n_examples: int = 3, max_chars: int = 1200):
    """Markdown with the same few eval questions at the first, middle and last logged eval step."""
    path = run_dir / "generations.jsonl"
    if not path.exists():
        return
    by_step = defaultdict(list)
    for line in path.open():
        if line.strip():
            g = json.loads(line)
            by_step[g["step"]].append(g)
    steps = sorted(by_step)
    picked = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
    lines = [f"# Rollouts over time: {run_dir.name}", "",
             f"Same {n_examples} eval questions (T=1 samples) at GRPO steps {picked}.", ""]
    for q in range(n_examples):
        question = by_step[picked[0]][q]["prompt"].split("User:", 1)[-1].split("Assistant:", 1)[0].strip()
        lines += [f"## Question {q + 1} (ground truth: {by_step[picked[0]][q]['ground_truth']})", "", f"> {question}", ""]
        for s in picked:
            g = by_step[s][q]
            resp = g["response"] if len(g["response"]) <= max_chars else g["response"][:max_chars] + " [...]"
            lines += [f"**step {s}** — reward {g['reward']:.0f}, format {g['format_reward']:.0f}, "
                      f"{g['response_length']} tokens, entropy {g['mean_token_entropy']:.2f}", "",
                      "```", resp.strip(), "```", ""]
    out = run_dir / "rollouts_over_time.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main(
    runs: list[str] = typer.Argument(..., help="run names under output_dir"),
    name: str = typer.Option("grpo", help="prefix for the combined plots"),
    output_dir: str = "outputs/grpo",
    tokenizer_name: str = "Qwen/Qwen2.5-Math-1.5B",
    window: int = 5,
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    out = Path(output_dir)
    run_dirs = {r: out / r for r in runs if (out / r / "metrics.jsonl").exists()}
    if not run_dirs:
        raise SystemExit("no runs with metrics.jsonl found")

    plot_eval_reward(run_dirs, out / f"{name}_eval_reward.png")
    plot_metrics(run_dirs, out / f"{name}_metrics.png", tokenizer, window)
    for run_dir in run_dirs.values():
        write_rollouts_over_time(run_dir)

    print(f"\n{'run':<24} {'steps':>6} {'last_eval':>9} {'best_eval':>9} {'last_fmt':>8} {'minutes':>8}")
    for label, run_dir in run_dirs.items():
        ev = load_split(run_dir, "eval")
        ro = load_split(run_dir, "rollout")
        if not ev:
            continue
        print(f"{label:<24} {ro[-1]['step'] if ro else 0:>6} {ev[-1]['answer_reward']:>9.3f} "
              f"{max(r['answer_reward'] for r in ev):>9.3f} {ev[-1]['format_reward']:>8.3f} "
              f"{ev[-1].get('wall_clock_s', 0) / 60:>8.1f}")


if __name__ == "__main__":
    typer.run(main)
