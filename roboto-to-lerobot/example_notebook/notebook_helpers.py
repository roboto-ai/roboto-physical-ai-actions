"""Plumbing helpers for the demo notebook.

Kept out of the notebook itself so each cell can stay focused on the
Roboto SDK calls it is meant to demonstrate.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import pathlib
import tempfile
import time
import webbrowser


def partition_events_lpt(events, n: int) -> list[list]:
    """LPT bin-pack ``events`` into ``n`` shards balanced by event duration.

    Mirrors ``_partition_shards`` in the action's main.py: sort events
    heaviest-first by ``end_time - start_time`` (frame-count proxy when
    fps is constant), greedy-assign each to the shard with the smallest
    running total. Returns one list of events per shard.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    shards: list[list] = [[] for _ in range(n)]
    # Min-heap of (running_total, shard_idx); ties broken by shard_idx.
    heap = [(0, i) for i in range(n)]
    heapq.heapify(heap)
    ordered = sorted(
        events, key=lambda e: int(e.end_time) - int(e.start_time), reverse=True
    )
    for event in ordered:
        weight = int(event.end_time) - int(event.start_time)
        total, idx = heapq.heappop(heap)
        shards[idx].append(event)
        heapq.heappush(heap, (total + weight, idx))
    return shards


def wait_all_invocations(invocations, timeout: float = 7200, poll_interval: float = 10):
    """Block until every invocation in ``invocations`` reaches a terminal status.

    Uses one thread per invocation so per-invocation polling overlaps. Returns
    the list of final ``InvocationStatus`` values in input order. Raises
    ``TimeoutError`` (from ``roboto.waiters``) if any invocation does not
    finish within ``timeout`` seconds.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(invocations)) as pool:
        futures = [
            pool.submit(iv.wait_for_terminal_status, timeout=timeout, poll_interval=poll_interval)
            for iv in invocations
        ]
        for fut in futures:
            fut.result()
    return [iv.current_status for iv in invocations]


def tail_until_done(iv, poll_interval: float = 3.0):
    """Stream an invocation's logs to stdout until it reaches a terminal status.

    ``Invocation.wait_for_terminal_status`` polls silently; calling
    ``stream_logs`` in parallel and printing each ``LogRecord`` surfaces the
    action's INFO logs inline under the cell. CloudWatch-side lag is ~5–15 s.
    """
    last_read = None

    def drain():
        nonlocal last_read
        gen = iv.stream_logs(last_read=last_read)
        while True:
            try:
                rec = next(gen)
            except StopIteration as stop:
                last_read = stop.value
                return
            ts = rec.timestamp.strftime("%H:%M:%S")
            print(f"[{ts}] {rec.log}", flush=True)

    while True:
        drain()
        iv.refresh()
        if iv.reached_terminal_status:
            drain()  # final flush: catch lines emitted after the last poll
            return iv.current_status
        time.sleep(poll_interval)


def download_and_open_audit_report(dataset, invocation_id: str) -> pathlib.Path:
    """Download an audit run's artifacts to a temp dir and open the HTML locally.

    The Jupyter/VS Code output iframe is a sandboxed origin that can't resolve
    the HTML's relative ``contract.yaml`` download link, so the report is
    opened in the system browser instead of displayed inline.
    """
    report_dir = pathlib.Path(tempfile.mkdtemp(prefix="audit-"))
    dataset.download_files(
        report_dir,
        include_patterns=[f"audit_reports/{invocation_id}/**"],
    )
    html_candidates = list(report_dir.rglob("audit_report.html"))
    if not html_candidates:
        dataset.download_files(report_dir, include_patterns=["audit_reports/**"])
        html_candidates = list(report_dir.rglob("audit_report.html"))
    html_path = max(html_candidates, key=lambda p: p.stat().st_mtime).resolve()
    print(f"Audit report: {html_path}")
    webbrowser.open(html_path.as_uri())
    return html_path
