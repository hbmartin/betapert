"""Generate the PDF-shapes figure embedded in the README.

Run with: uv run python docs/generate_pdf_shapes.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from betapert import mpert

OUT = Path(__file__).parent / "pdf_shapes.svg"


def main() -> None:
    x = np.linspace(0, 10, 500)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lambd, color in [(1, "#9ecae1"), (2, "#6baed6"), (4, "#2171b5"), (8, "#08306b")]:
        label = f"lambd={lambd}" + (" (classic PERT)" if lambd == 4 else "")
        ax.plot(x, mpert.pdf(x, 0, 3, 10, lambd), color=color, label=label)
    ax.set_title("Modified PERT(mini=0, mode=3, maxi=10) densities")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
