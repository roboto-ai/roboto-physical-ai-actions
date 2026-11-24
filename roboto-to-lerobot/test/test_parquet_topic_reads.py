"""Tests that the action can read topics Roboto stores as Parquet, such as derived topics.

The Roboto SDK reads such a topic over HTTP from a signed URL; here the URL
points at a local HTTP server.
"""

from __future__ import annotations

import functools
import http.server
import threading

import pandas as pd
import pytest
from roboto.formats.parquet.fetch import parquet_file_from_url


@pytest.fixture
def served_parquet(tmp_path):
    """A small topic as a Parquet file, and the URL of a local server serving it."""
    df = pd.DataFrame({"timestamp": [1, 2, 3], "x": [0.5, 1.5, 2.5]})
    df.to_parquet(tmp_path / "topic.parquet")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield df, f"http://127.0.0.1:{server.server_port}/topic.parquet"
    server.shutdown()


def test_parquet_topic_reads_over_http(served_parquet):
    """The SDK's reader for Parquet-backed topics works in the action's environment."""
    df, url = served_parquet

    table = parquet_file_from_url(url).read()

    pd.testing.assert_frame_equal(table.to_pandas(), df)
