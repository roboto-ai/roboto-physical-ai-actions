from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from ..core.types import QualityReport

_METRIC_EXPLANATIONS: dict[str, str] = {
    "autocorrelation": (
        "Per-dimension action autocorrelation r(k). A signal's value at time t "
        "predicts its value at t+k — slower decay ⇒ more redundant frames and "
        "fewer effective samples. The primary panel plots actions only, "
        "matching the binning the LeRobot dataset visualizer's Action "
        "Autocorrelation panel uses for visual familiarity; the extended "
        "state+action τ_int/τ_half summary below goes further than that "
        "visualizer computes. The dotted r=0.5 line marks τ_half (lags where "
        "the signal has 'half-forgotten' itself)."
    ),
    "state_action_alignment": (
        "Pearson correlation of Δaction(t) vs Δstate(t+τ) per paired dim, "
        "comparable to the LeRobot dataset visualizer's State-Action Temporal "
        "Alignment panel. Peaks near τ=0 indicate the commanded action and "
        "the sensed state change together; peaks at |τ|≥2 suggest the "
        "dataset has a timestamp offset between observations and actions. "
        "Curves aggregate paired dims as max/mean/min envelopes."
    ),
    "speed_distribution": (
        "Per-episode mean step size: mean_t ||action(t)−action(t−1)||₂. The "
        "histogram across episodes exposes how consistent demonstrator speed "
        "is; high coefficient-of-variation means pace varies a lot between "
        "demos (harder to learn a single policy). The extended SPARC/jerk "
        "stats below go beyond what the LeRobot dataset visualizer's "
        "equivalent panel computes (its 'jerk' is a first-difference of "
        "velocity, not true jerk)."
    ),
    "cross_episode_variance": (
        "For each (action dim, normalized time) cell, the standard deviation "
        "across episodes (square-root remapped for display), comparable to "
        "the LeRobot dataset visualizer's Cross-Episode Action Variance "
        "panel. Bright bands mark time regions where episodes diverge — "
        "usually where operators make independent choices."
    ),
    "action_velocity": (
        "Pooled Δaction distribution per dim and a ranked list of the jerkiest "
        "episodes (highest mean |Δa|). Jerky episodes often correspond to "
        "teleop glitches or unsmoothed replays. There is no equivalent ranked "
        "view in the LeRobot dataset visualizer."
    ),
    "filtering_flags": (
        "Dataset-level flags that mark an episode for removal: length outlier "
        "(|T − median|>3·MAD), zero-variance dim, etc. The cleanup command "
        "below this section is ready to paste into `lerobot-edit-dataset`."
    ),
    "effective_sample_size": (
        "ESS = T / τ_int, where τ_int is the Sokal integrated autocorrelation "
        "time. A stream with τ_int=30 has only T/30 independent samples — "
        "declared frame counts overstate training signal when data is smooth. "
        "This is a Roboto-added metric with no equivalent in the LeRobot "
        "dataset visualizer."
    ),
    "effective_dimensionality": (
        "PCA participation ratio PR = (Σλ)²/Σλ². PR near the full dim means "
        "variation is spread across all channels; PR << dim means a few "
        "channels dominate and the rest are nearly redundant."
    ),
    "state_coverage": (
        "Per-state-dim (p99−p01)/declared_range. Values near 1.0 mean the "
        "dataset exercises the full declared operating envelope; values near 0 "
        "mean the robot stayed in a narrow band (the declared range will "
        "over-promise coverage to anyone training on this data)."
    ),
    "state_coverage__no_declared": (
        "Not computed for this dataset: info.json declares no min/max for the "
        "state feature, so there is no denominator to express coverage as a "
        "fraction of a declared operating envelope. Add stats.min and "
        "stats.max to the feature (or regenerate meta/stats.json) and re-run "
        "the audit to see this plot."
    ),
}

_CSS = """
:root {
    --bg: #ffffff;
    --panel: #f9fafb;
    --panel-hi: #f3f4f6;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --heading: #1f2937;
    --accent: #dc2626;
    --accent-bg: #fef2f2;
    --accent-border: #fecaca;
    --nav-bg: #111827;
    --nav-fg: #f9fafb;
    --nav-accent: #dc2626;
    --cli-bg: #1f2937;
    --cli-fg: #e5e7eb;
    --link: #2563eb;
    --plot-accent: #f97316;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
}
.banner {
    background: var(--nav-bg);
    color: var(--nav-fg);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid var(--nav-accent);
}
.banner .brand {
    display: flex; align-items: baseline; gap: .75rem;
}
.banner .logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem; height: 1.6rem;
    background: var(--nav-accent);
    color: #fff;
    border-radius: 6px;
    font-weight: 700;
    font-size: .95rem;
}
.banner h1 {
    font-size: 1.1rem;
    margin: 0;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.banner .mode {
    font-size: .8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.banner .source code {
    background: rgba(255,255,255,0.08);
    color: var(--nav-fg);
    padding: .2rem .5rem;
    border-radius: 4px;
    font-size: .8rem;
}
.content {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
}
h2 {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--heading);
    margin: 2rem 0 .75rem 0;
}
h3 {
    font-size: .95rem;
    font-weight: 600;
    color: var(--heading);
    margin: 1.25rem 0 .5rem 0;
}
.header {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .8rem;
    margin: 1rem 0 1.5rem 0;
}
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .85rem 1rem;
}
.card .n {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--heading);
    line-height: 1.2;
}
.card .l {
    font-size: .75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: .2rem;
}
.card.alert {
    background: var(--accent-bg);
    border-color: var(--accent-border);
}
.card.alert .n { color: var(--accent); }
.flag-block {
    background: var(--accent-bg);
    border: 1px solid var(--accent-border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0 1.5rem 0;
}
.flag-block h2 {
    margin-top: 0;
    color: var(--accent);
}
.cli {
    background: var(--cli-bg);
    color: var(--cli-fg);
    padding: .85rem 1rem;
    border-radius: 6px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .8rem;
    overflow-x: auto;
    white-space: pre;
    margin-top: .75rem;
}
table {
    border-collapse: collapse;
    width: 100%;
    font-size: .8rem;
    margin: .5rem 0 1rem 0;
    background: var(--bg);
}
th, td {
    border: 1px solid var(--border);
    padding: .35rem .55rem;
    text-align: left;
    vertical-align: top;
}
th {
    background: var(--panel);
    font-weight: 600;
    color: var(--heading);
    position: sticky;
    top: 0;
}
tr.flagged { background: var(--accent-bg); }
tr.flagged td:first-child { color: var(--accent); font-weight: 600; }
img {
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
    margin: .5rem 0;
    background: var(--bg);
}
.small { font-size: .8rem; color: var(--muted); }
details > summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--heading);
    padding: .3rem 0;
    user-select: none;
}
details > summary:hover { color: var(--accent); }
pre {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: .7rem .85rem;
    overflow-x: auto;
    font-size: .78rem;
    border-radius: 4px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--text);
}
code {
    background: var(--panel-hi);
    padding: .1rem .35rem;
    border-radius: 3px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .85em;
}
.metric-section {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.metric-section h3 {
    margin-top: 0;
    padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
}
.metric-section .explain {
    font-size: .82rem;
    color: var(--muted);
    margin: .4rem 0 .6rem 0;
    line-height: 1.45;
}
.err { color: var(--accent); font-size: .8rem; }
"""


def render_html_report(
    report: QualityReport,
    plots: dict[str, Path],
    out_path: Path,
    run_dir: Path | None = None,
) -> Path:
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>Training dataset audit</title>")
    parts.append(f"<style>{_CSS}</style></head><body>")
    parts.append(_banner(report))
    parts.append("<div class='content'>")
    parts.append(_header(report))
    parts.append(_flag_block(report))
    parts.append(_contract_section(report, run_dir))
    parts.append(_metrics_sections(report, plots))
    parts.append(_raw_json(report))
    parts.append("</div></body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def _contract_section(report: QualityReport, run_dir: Path | None) -> str:
    ref = report.contract
    if ref is None:
        return ""
    yaml_text = ""
    if run_dir is not None:
        src = run_dir / ref.filename
        if src.is_file():
            try:
                yaml_text = src.read_text(encoding="utf-8")
            except OSError:
                yaml_text = ""
    body = (
        f"<pre>{html.escape(yaml_text)}</pre>"
        if yaml_text
        else "<p class='small'><em>contract file not available for inline preview</em></p>"
    )
    return (
        "<h2>Contract</h2>"
        "<p class='small'>"
        f"Archived as <code>{html.escape(ref.filename)}</code> "
        f"(sha256 <code>{html.escape(ref.sha256[:12])}…</code>). "
        f"<a href='{html.escape(ref.filename)}' download>Download contract.yaml</a>"
        "</p>"
        "<details><summary>view contract.yaml</summary>"
        f"{body}</details>"
    )


def _banner(report: QualityReport) -> str:
    ds = report.dataset_summary
    return (
        "<div class='banner'>"
        "<div class='brand'>"
        "<span class='logo'>R</span>"
        "<h1>Training dataset audit</h1>"
        f"<span class='mode'>{html.escape(ds.mode)}</span>"
        "</div>"
        "<div class='source'>"
        f"<code>{html.escape(report.source.kind)}:{html.escape(report.source.identifier)}</code>"
        "</div>"
        "</div>"
    )


def _header(report: QualityReport) -> str:
    ds = report.dataset_summary
    n_flagged = ds.n_flagged_episodes
    ess = f"{ds.ess_total:.0f}" if ds.ess_total is not None else "—"
    alert = "alert" if n_flagged > 0 else ""
    header = (
        f"<div class='header'>"
        f"<div class='card'><div class='n'>{ds.n_episodes}</div><div class='l'>episodes</div></div>"
        f"<div class='card'><div class='n'>{ds.n_frames:,}</div><div class='l'>frames (declared)</div></div>"
        f"<div class='card'><div class='n'>{ess}</div><div class='l'>effective sample size</div></div>"
        f"<div class='card {alert}'><div class='n'>{n_flagged}</div>"
        f"<div class='l'>flagged episodes</div></div>"
        f"</div>"
    )
    if ds.low_effective_dim:
        # Dataset-level PCA verdict (see effective_dimensionality section) —
        # deliberately not one of the per-episode flagged-episode cards above.
        header += (
            "<p class='small' style='color:var(--accent);'>"
            "Dataset-level: PCA participation ratio is below 0.3 — variation "
            "across all episodes is concentrated in a few channels "
            "(see the effective_dimensionality section below)."
            "</p>"
        )
    return header


def _flag_block(report: QualityReport) -> str:
    if not report.flagged_episodes:
        return "<p class='small'><em>No episodes flagged.</em></p>"
    rows = "".join(
        f"<tr class='flagged'><td>{fe.episode_index}</td>"
        f"<td>{html.escape(', '.join(fe.flags))}</td>"
        f"<td>{html.escape(fe.reason)}</td></tr>"
        for fe in report.flagged_episodes
    )
    return (
        "<div class='flag-block'>"
        "<h2>Flagged episodes</h2>"
        "<table><tr><th>episode</th><th>flags</th><th>reason</th></tr>"
        f"{rows}</table></div>"
    )


def _metrics_sections(report: QualityReport, plots: dict[str, Path]) -> str:
    out: list[str] = ["<h2>Metrics</h2>"]
    for name, metric in report.metrics.items():
        out.append("<div class='metric-section'>")
        out.append(f"<h3>{html.escape(name)}</h3>")
        explanation = _explanation_for(name, metric)
        if explanation:
            out.append(f"<p class='explain'>{html.escape(explanation)}</p>")
        if name in plots:
            b64 = base64.b64encode(plots[name].read_bytes()).decode("ascii")
            out.append(f"<img src='data:image/png;base64,{b64}' alt='{name}'/>")
        if metric.errors:
            errs = "<br/>".join(html.escape(e) for e in metric.errors)
            out.append(f"<p class='err'>errors: {errs}</p>")
        out.append("<details><summary>per-dataset</summary>")
        out.append(_table_from_kv(metric.per_dataset))
        out.append("</details>")
        if metric.per_episode:
            out.append("<details><summary>per-episode</summary>")
            out.append(_table_from_dicts(metric.per_episode))
            out.append("</details>")
        out.append("</div>")
    return "\n".join(out)


def _raw_json(report: QualityReport) -> str:
    js = report.model_dump_json(indent=2)
    return (
        "<details><summary>raw report.json</summary>"
        f"<pre>{html.escape(js)}</pre></details>"
    )


def _table_from_dicts(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body_rows = []
    for r in rows:
        cells = "".join(
            f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in cols
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body_rows)}</table>"


def _explanation_for(name: str, metric) -> str | None:
    if name == "state_coverage":
        state = (metric.per_dataset or {}).get("state") or {}
        if not state.get("coverage_fraction"):
            return _METRIC_EXPLANATIONS.get("state_coverage__no_declared")
    return _METRIC_EXPLANATIONS.get(name)


def _flatten_kv(obj, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a nested dict into dot-path keyed (key, value) pairs. Lists of
    scalars stay inline; lists of dicts get indexed."""
    out: list[tuple[str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.extend(_flatten_kv(v, key))
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for i, item in enumerate(v):
                    out.extend(_flatten_kv(item, f"{key}[{i}]"))
            else:
                out.append((key, v))
    else:
        out.append((prefix or "value", obj))
    return out


def _table_from_kv(obj) -> str:
    rows = _flatten_kv(obj)
    if not rows:
        return "<p class='small'><em>empty</em></p>"
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt(v))}</td></tr>"
        for k, v in rows
    )
    return f"<table><tr><th>key</th><th>value</th></tr>{body}</table>"


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if abs(v) >= 1000 or (0 < abs(v) < 0.01):
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str)
    return str(v)
