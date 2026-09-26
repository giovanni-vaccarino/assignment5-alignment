import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, just write PNGs
import matplotlib.pyplot as plt


class LocalLogger:
    """
    Minimal local replacement for wandb.

    log("train", step, {"loss": ...}) appends a line to <run_dir>/metrics.jsonl and keeps the
    history in memory. plot() redraws <run_dir>/plots/<split>.png with one subplot per metric,
    each plotted against its own step axis (like wandb's train_step / eval_step).
    Open the PNGs in VS Code: they refresh when the file changes.
    """

    def __init__(self, run_dir: str | Path, config: dict | None = None):
        self.run_dir = Path(run_dir)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.history: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def log(self, split: str, step: int, metrics: dict):
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"split": split, "step": step, **clean}) + "\n")
        for k, v in clean.items():
            self.history[split][k].append((step, v))

    def plot(self, split: str | None = None):
        for s in [split] if split is not None else list(self.history):
            metrics = self.history[s]
            if not metrics:
                continue
            n = len(metrics)
            cols = min(3, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
            for ax, (name, points) in zip(axes.flat, sorted(metrics.items())):
                steps, values = zip(*points)
                ax.plot(steps, values, marker="o" if len(points) < 30 else None, markersize=3)
                ax.set_title(f"{s}/{name}")
                ax.set_xlabel(f"{s}_step")
                ax.grid(alpha=0.3)
            for ax in list(axes.flat)[n:]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(self.run_dir / "plots" / f"{s}.png", dpi=100)
            plt.close(fig)
