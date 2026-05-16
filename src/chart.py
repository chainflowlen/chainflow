"""
src/chart.py — 7-day stablecoin CEX netflow line chart.

Design spec: black background · blue/green lines per token · no grid · zero reference line.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

_COLORS = {
    "USDC": "#4F9BFF",   # blue
    "USDT": "#26C281",   # green
}
_BG = "#0D0D0D"
_AXIS_COLOR = "#AAAAAA"
_SPINE_COLOR = "#444444"
_ZERO_LINE_COLOR = "#555555"


def plot_netflow_chart(
    daily_df: pd.DataFrame,
    output_path: str = "output/netflow_7d.png",
    title: str = "7-Day Stablecoin CEX Netflow",
) -> str:
    """
    Render a 7-day netflow line chart and save to `output_path`.

    Args:
        daily_df:    DataFrame with columns [date, token, inflow, outflow, netflow].
                     `date` may be date objects or ISO strings.
        output_path: Destination PNG path (parent directories are created).
        title:       Chart title string.

    Returns:
        Absolute path to the saved PNG.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    if daily_df.empty:
        ax.text(
            0.5, 0.5, "No data available",
            ha="center", va="center", color="white", transform=ax.transAxes,
        )
    else:
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        if "token" in daily_df.columns:
            for token in sorted(daily_df["token"].unique()):
                subset = daily_df[daily_df["token"] == token].sort_values("date")
                color = _COLORS.get(token, "#FFFFFF")
                ax.plot(
                    subset["date"],
                    subset["netflow"],
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=token,
                )
            ax.legend(
                frameon=False,
                labelcolor="white",
                fontsize=10,
                loc="upper left",
            )
        else:
            daily_df = daily_df.sort_values("date")
            ax.plot(
                daily_df["date"],
                daily_df["netflow"],
                color=_COLORS["USDC"],
                linewidth=2,
                marker="o",
                markersize=4,
            )

    # Zero reference line
    ax.axhline(0, color=_ZERO_LINE_COLOR, linewidth=0.8, linestyle="--")

    # Titles & labels
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel("Date", color=_AXIS_COLOR, fontsize=10)
    ax.set_ylabel("Netflow (USD)", color=_AXIS_COLOR, fontsize=10)

    # Tick styling
    ax.tick_params(colors="white", which="both")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=30, ha="right")

    # Y-axis in millions
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x / 1_000_000:.1f}M")
    )

    # Spines
    for side in ("bottom", "left"):
        ax.spines[side].set_color(_SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    return str(Path(output_path).resolve())
