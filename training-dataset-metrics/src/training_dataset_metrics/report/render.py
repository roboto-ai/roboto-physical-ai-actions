from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ..core.types import QualityReport

# HF visualizer Recharts palette — applied to plot data so the shapes users
# see here are visually recognizable from HF's own Action Insights panel.
_PALETTE = (
    "#f97316",  # orange
    "#3b82f6",  # blue
    "#22c55e",  # green
    "#ef4444",  # red
    "#a855f7",  # violet
    "#eab308",  # yellow
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f59e0b",  # amber
    "#6366f1",  # indigo
    "#84cc16",  # lime
)

# Plot chrome (fig/axes background, grid, spines, labels) follows Roboto's
# product palette for visual continuity with the surrounding HTML page.
_CHROME = {
    "bg": "#ffffff",
    "grid": "#e5e7eb",
    "axis": "#9ca3af",
    "label": "#374151",
    "muted": "#6b7280",
}

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "hf_blue_teal_orange", ["#3b82f6", "#14b8a6", "#f97316"]
)


def render_all_plots(report: QualityReport, out_dir: Path) -> dict[str, Path]:
    """Render one PNG per metric that has something visualizable.

    Returns {metric_name: path}. Metrics with no drawable artifacts are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}

    renderers = {
        "autocorrelation": _render_autocorrelation,
        "state_action_alignment": _render_alignment,
        "speed_distribution": _render_speed,
        "cross_episode_variance": _render_variance,
        "action_velocity": _render_action_velocity,
        "effective_sample_size": _render_ess,
        "effective_dimensionality": _render_effective_dim,
        "state_coverage": _render_coverage,
        "filtering_flags": _render_filtering,
    }

    for name, metric in report.metrics.items():
        renderer = renderers.get(name)
        if renderer is None:
            continue
        path = out_dir / f"{name}.png"
        try:
            fig = renderer(metric.per_episode, metric.per_dataset, metric.artifacts)
        except Exception:
            continue
        if fig is None:
            continue
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=_CHROME["bg"])
        plt.close(fig)
        rendered[name] = path

    return rendered


# ---------- styling helpers ----------


def _apply_style(ax: plt.Axes, *, title: str | None = None) -> None:
    """Apply shared plot chrome: white bg, muted grid + spines, HF color cycle."""
    ax.set_facecolor(_CHROME["bg"])
    ax.grid(True, color=_CHROME["grid"], lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_CHROME["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_CHROME["label"], labelsize=9)
    ax.xaxis.label.set_color(_CHROME["label"])
    ax.yaxis.label.set_color(_CHROME["label"])
    ax.set_prop_cycle(color=list(_PALETTE))
    if title:
        ax.set_title(title, color=_CHROME["label"], fontsize=11, loc="left")


def _empty(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color=_CHROME["muted"], fontsize=10)
    ax.axis("off")


def _draw_half_line(ax: plt.Axes) -> None:
    """r=0.5 reference line — bolder than the grid, labeled, placed above
    the data so it stays visible in a busy plot."""
    ax.axhline(
        0.5,
        color=_CHROME["muted"],
        lw=1.4,
        ls=(0, (4, 3)),
        alpha=0.9,
        zorder=3,
        label="r=0.5",
    )


def _two_panel_fig() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax_p, ax_e) = plt.subplots(
        1, 2,
        figsize=(12.5, 4.4),
        gridspec_kw={"width_ratios": [1.25, 1]},
        facecolor=_CHROME["bg"],
    )
    return fig, ax_p, ax_e


# ---------- primary (LeRobot-visualizer-comparable where noted) + extended renderers ----------


def _render_autocorrelation(
    per_episode: list[dict], per_dataset: dict, artifacts: dict
) -> Any:
    primary = (artifacts or {}).get("primary") or {}
    extended = (artifacts or {}).get("extended") or {}

    fig, ax_p, ax_e = _two_panel_fig()

    # Primary: actions-only ACF, HF divisor.
    primary_curves: dict = primary.get("curves_by_episode") or {}
    _apply_style(ax_p, title="Action autocorrelation")
    ax_p.set_xlabel("Lag (steps)")
    ax_p.set_ylabel("Autocorrelation")
    _draw_half_line(ax_p)
    drew = False
    for _ep_idx, payload in primary_curves.items():
        lags = payload.get("lags") or []
        curves = payload.get("curves") or []
        for ci, curve in enumerate(curves):
            ax_p.plot(lags, curve, lw=1.2, alpha=0.85,
                      color=_PALETTE[ci % len(_PALETTE)])
            drew = True
    if not drew:
        _empty(ax_p, "no data")

    # Extended: state+action per-role curves.
    ext_curves: dict = extended.get("curves_by_episode") or {}
    _apply_style(ax_e, title="Extended — state & action τ curves")
    ax_e.set_xlabel("Lag (frames)")
    ax_e.set_ylabel("r")
    _draw_half_line(ax_e)
    ext_drew = False
    for _ep_idx, per_role in ext_curves.items():
        for role_i, role in enumerate(("state", "action")):
            payload = per_role.get(role)
            if not payload:
                continue
            lags = payload.get("lags") or []
            for ci, curve in enumerate(payload.get("curves") or []):
                ax_e.plot(
                    lags, curve, lw=0.8, alpha=0.45,
                    color=_PALETTE[(role_i * 5 + ci) % len(_PALETTE)],
                )
                ext_drew = True
    if not ext_drew:
        _empty(ax_e, "no extended data")
    return fig


def _render_alignment(per_episode, per_dataset, artifacts):
    primary = (artifacts or {}).get("primary") or {}
    extended = (artifacts or {}).get("extended") or {}

    fig, ax_p, ax_e = _two_panel_fig()

    # Primary: max/mean/min envelope across paired dims.
    _apply_style(ax_p, title="State↔Action temporal alignment (LeRobot-comparable)")
    ax_p.set_xlabel("Lag (steps)")
    ax_p.set_ylabel("Pearson r")
    ax_p.axhline(0, color=_CHROME["axis"], lw=0.6)
    ax_p.axvline(0, color=_CHROME["axis"], lw=0.6, ls="--")
    primary_by_ep: dict = primary.get("by_episode") or {}
    drew = False
    for _ep_idx, payload in primary_by_ep.items():
        lags = payload.get("lags") or []
        if not lags:
            continue
        ax_p.plot(lags, payload.get("max") or [], color=_PALETTE[0],
                  lw=1.2, alpha=0.85, label="max" if not drew else None)
        ax_p.plot(lags, payload.get("mean") or [], color=_PALETTE[1],
                  lw=1.2, alpha=0.85, label="mean" if not drew else None)
        ax_p.plot(lags, payload.get("min") or [], color=_PALETTE[2],
                  lw=1.2, alpha=0.85, label="min" if not drew else None)
        drew = True
    if drew:
        ax_p.legend(loc="best", fontsize=9, frameon=False, labelcolor=_CHROME["label"])
    else:
        _empty(ax_p, "no paired dims")

    # Extended: per-pair cross-correlation curves.
    ext_by_ep: dict = extended.get("by_episode") or {}
    _apply_style(ax_e, title="Extended — per-pair cross-correlation")
    ax_e.set_xlabel("Lag (frames)")
    ax_e.set_ylabel("xcorr")
    ax_e.axhline(0, color=_CHROME["axis"], lw=0.6)
    ax_e.axvline(0, color=_CHROME["axis"], lw=0.6, ls="--")
    ext_drew = False
    ci = 0
    for _ep_idx, payload in ext_by_ep.items():
        for curve in payload.get("pair_curves") or []:
            ax_e.plot(curve["lags"], curve["xcorr"], lw=0.8, alpha=0.5,
                      color=_PALETTE[ci % len(_PALETTE)])
            ext_drew = True
            ci += 1
    if not ext_drew:
        _empty(ax_e, "no pairs")
    return fig


def _render_speed(per_episode, per_dataset, artifacts):
    primary = (artifacts or {}).get("primary") or {}

    fig, ax_p, ax_e = _two_panel_fig()

    # Primary: HF per-episode L2-norm-of-Δa scalar histogram.
    verdict = primary.get("verdict") or per_dataset.get("verdict") or "?"
    _apply_style(ax_p, title=f"Demonstrator speed variance — {verdict}")
    ax_p.set_xlabel("Mean step size  mean_t ‖Δa‖₂")
    ax_p.set_ylabel("Episodes")
    hist = primary.get("hist") or {}
    bins = hist.get("bins") or []
    counts = hist.get("counts") or []
    if bins and counts:
        edges = np.asarray(bins, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        ax_p.bar(centers, counts, width=widths * 0.95, color=_PALETTE[0],
                 edgecolor=_CHROME["bg"], linewidth=0.5)
        med = primary.get("median")
        if med is not None and np.isfinite(med):
            ax_p.axvline(med, color=_PALETTE[3], lw=1.0, ls="--",
                         label=f"median={med:.4g}")
            ax_p.legend(loc="best", fontsize=9, frameon=False,
                        labelcolor=_CHROME["label"])
        ax_p.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        _empty(ax_p, "no episodes")

    # Extended: per-episode mean action speed scatter.
    _apply_style(ax_e, title="Extended — per-episode mean |Δa|·fps")
    ax_e.set_xlabel("Episode")
    ax_e.set_ylabel("Mean |Δa|·fps")
    ep_ids = [r.get("episode_index") for r in per_episode]
    means = [r.get("action_mean_speed") for r in per_episode]
    ids_x = [i for i, m in enumerate(means) if m is not None and np.isfinite(m)]
    vals = [means[i] for i in ids_x]
    labels = [str(ep_ids[i]) for i in ids_x]
    if vals:
        ax_e.bar(np.arange(len(vals)), vals, color=_PALETTE[1],
                 edgecolor=_CHROME["bg"], linewidth=0.5)
        ax_e.set_xticks(np.arange(len(vals)))
        ax_e.set_xticklabels(labels, rotation=60, fontsize=8)
    else:
        _empty(ax_e, "no data")
    return fig


def _render_variance(per_episode, per_dataset, artifacts):
    primary = (artifacts or {}).get("primary") or {}
    extended = (artifacts or {}).get("extended") or {}

    fig, ax_p, ax_e = _two_panel_fig()

    # Primary: (D × 50) cross-episode variance heatmap. Stored raw; sqrt
    # remap here matches HF's color-scaling behavior.
    matrix = np.asarray(primary.get("matrix") or [], dtype=float)
    dim_names = primary.get("dim_names") or []
    _apply_style(
        ax_p, title="Cross-episode action variance (LeRobot-comparable, √-color)"
    )
    ax_p.set_xlabel("Episode progress →")
    ax_p.set_ylabel("Action dim")
    if matrix.ndim == 2 and matrix.size > 0:
        display_matrix = np.sqrt(np.maximum(matrix, 0.0))
        im = ax_p.imshow(display_matrix, aspect="auto", cmap=_HEATMAP_CMAP,
                         interpolation="nearest")
        ax_p.set_yticks(np.arange(matrix.shape[0]))
        ax_p.set_yticklabels(dim_names or [f"{i}" for i in range(matrix.shape[0])],
                             fontsize=8)
        ax_p.set_xticks([])
        ax_p.grid(False)
        cbar = fig.colorbar(im, ax=ax_p, shrink=0.85)
        cbar.ax.tick_params(colors=_CHROME["label"], labelsize=8)
    else:
        _empty(ax_p, "no action data")

    # Extended: (episode × dim) normalized std heatmap grid.
    heatmaps: dict = extended.get("heatmaps") or {}
    kinds = list(heatmaps.keys())
    _apply_style(ax_e, title="Extended — per-episode std (normalized)")
    ax_e.grid(False)
    if not kinds:
        _empty(ax_e, "no extended heatmaps")
    else:
        # Show the first available one; the full grid lives in the HTML tables.
        kind = kinds[0]
        payload = heatmaps[kind]
        ext_matrix = np.asarray(payload["matrix"], dtype=float)
        ep_indices = payload.get("episode_index") or []
        ax_e.set_xlabel(f"{kind} dim")
        ax_e.set_ylabel("Episode")
        if ext_matrix.size > 0:
            im2 = ax_e.imshow(ext_matrix, aspect="auto", cmap=_HEATMAP_CMAP,
                              interpolation="nearest")
            if ep_indices:
                n_rows = ext_matrix.shape[0]
                # Keep tick density reasonable on big datasets.
                step = max(1, n_rows // 20)
                tick_pos = np.arange(0, n_rows, step)
                ax_e.set_yticks(tick_pos)
                ax_e.set_yticklabels([str(int(ep_indices[i])) for i in tick_pos],
                                     fontsize=8)
            else:
                ax_e.yaxis.set_major_locator(MaxNLocator(integer=True))
            n_cols = ext_matrix.shape[1]
            ax_e.set_xticks(np.arange(n_cols))
            ax_e.set_xticklabels([str(i) for i in range(n_cols)], fontsize=8)
            cbar = fig.colorbar(im2, ax=ax_e, shrink=0.85)
            cbar.ax.tick_params(colors=_CHROME["label"], labelsize=8)
        else:
            _empty(ax_e, "empty matrix")
    return fig


# ---------- extended-only renderers (no LeRobot visualizer equivalent) ----------


def _render_action_velocity(per_episode, per_dataset, artifacts):
    per_dim = (artifacts or {}).get("delta_hist_per_dim") or []
    top_jerky = per_dataset.get("top_jerky_episodes", []) or []
    if not per_dim and not top_jerky:
        return None
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4), facecolor=_CHROME["bg"])
    _apply_style(ax_a, title="Δa distribution per action dim (30 bins)")
    _apply_style(ax_b, title="Top-jerky episodes")
    drew = False
    for entry in per_dim:
        bins = entry.get("bins") or []
        counts = entry.get("counts") or []
        d = entry.get("dim_index", 0)
        if not bins or not counts or not any(counts):
            continue
        edges = np.asarray(bins, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_a.plot(centers, counts, lw=1.0, alpha=0.85,
                  color=_PALETTE[d % len(_PALETTE)], label=f"dim {d}")
        drew = True
    if drew:
        ax_a.set_xlabel("Δa (native units)")
        ax_a.set_ylabel("Count (pooled frames)")
        if len(per_dim) <= 12:
            ax_a.legend(fontsize=8, frameon=False, loc="best",
                        labelcolor=_CHROME["label"], ncol=2)
    else:
        _empty(ax_a, "no Δa samples")
    if top_jerky:
        labels = [f"ep{e['episode_index']}" for e in top_jerky]
        values = [e["mean_abs_delta"] for e in top_jerky]
        ax_b.barh(labels[::-1], values[::-1], color=_PALETTE[3],
                  edgecolor=_CHROME["bg"], linewidth=0.5)
        ax_b.set_xlabel("Mean |Δa|")
    else:
        _empty(ax_b, "no jerky episodes")
    return fig


def _render_ess(per_episode, per_dataset, artifacts):
    state_med = [r.get("state_ess_median") for r in per_episode]
    action_med = [r.get("action_ess_median") for r in per_episode]
    ep_ids = [r.get("episode_index") for r in per_episode]
    state_med = [v if v is not None and np.isfinite(v) else np.nan for v in state_med]
    action_med = [v if v is not None and np.isfinite(v) else np.nan for v in action_med]
    if not any(np.isfinite(v) for v in state_med + action_med):
        return None
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=_CHROME["bg"])
    _apply_style(ax, title="Effective sample size per episode (median across channels)")
    idx = np.arange(len(ep_ids))
    width = 0.4
    ax.bar(idx - width / 2, state_med, width, label="state ESS median",
           color=_PALETTE[1], edgecolor=_CHROME["bg"], linewidth=0.5)
    ax.bar(idx + width / 2, action_med, width, label="action ESS median",
           color=_PALETTE[0], edgecolor=_CHROME["bg"], linewidth=0.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(e) for e in ep_ids], rotation=60, fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ESS (frames)")
    ax.legend(frameon=False, fontsize=9, labelcolor=_CHROME["label"])
    return fig


def _render_effective_dim(per_episode, per_dataset, artifacts):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=_CHROME["bg"])
    _apply_style(ax, title="PCA explained variance (cumulative)")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative explained variance")
    drew = False
    for i, role in enumerate(("state", "action")):
        payload = per_dataset.get(role) or {}
        evr = payload.get("explained_variance_ratio")
        if not evr:
            continue
        cum = np.cumsum(evr)
        ax.plot(np.arange(1, len(cum) + 1), cum, "o-", color=_PALETTE[i],
                label=f"{role} (PR={payload.get('pr'):.2f})")
        drew = True
    if drew:
        ax.axhline(0.95, color=_CHROME["axis"], lw=0.6, ls="--")
        ax.legend(frameon=False, fontsize=9, labelcolor=_CHROME["label"])
    else:
        _empty(ax, "no data")
    return fig


def _render_coverage(per_episode, per_dataset, artifacts):
    state = per_dataset.get("state") or {}
    frac = state.get("coverage_fraction")
    if not frac:
        # Without a declared range there is nothing to normalize by — skip
        # the plot entirely; the HTML section's explanation covers why.
        return None
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=_CHROME["bg"])
    _apply_style(ax, title="State coverage of declared range")
    clean = [float(x) if x is not None and np.isfinite(x) else 0.0 for x in frac]
    ax.bar(np.arange(len(clean)), clean, color=_PALETTE[1],
           edgecolor=_CHROME["bg"], linewidth=0.5)
    ax.axhline(1.0, color=_CHROME["axis"], lw=0.6, ls="--")
    ax.set_xlabel("State dim")
    ax.set_ylabel("(p99 − p01) / declared range")
    return fig


def _render_filtering(per_episode, per_dataset, artifacts):
    lengths = [r.get("n_frames") for r in per_episode]
    ep_ids = [r.get("episode_index") for r in per_episode]
    lengths = [v if v is not None else 0 for v in lengths]
    if not lengths:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=_CHROME["bg"])
    _apply_style(ax, title="Episode length distribution (outlier detection)")
    ax.bar(np.arange(len(ep_ids)), lengths, color=_PALETTE[1],
           edgecolor=_CHROME["bg"], linewidth=0.5)
    med = per_dataset.get("median_length", 0) or 0
    mad = per_dataset.get("length_mad", 1) or 1
    ax.axhline(med, color=_CHROME["axis"], ls="--", lw=0.8, label=f"median={med:.0f}")
    ax.axhline(med + 3 * mad, color=_PALETTE[3], ls=":", lw=0.8, label="±3·MAD")
    ax.axhline(max(0, med - 3 * mad), color=_PALETTE[3], ls=":", lw=0.8)
    ax.set_xticks(np.arange(len(ep_ids)))
    ax.set_xticklabels([str(e) for e in ep_ids], rotation=60, fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("n_frames")
    ax.legend(frameon=False, fontsize=9, labelcolor=_CHROME["label"])
    return fig
