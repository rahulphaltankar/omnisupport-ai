"""Consistent, small, headless figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030", "#718096", "#d69e2e", "#319795"]
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
