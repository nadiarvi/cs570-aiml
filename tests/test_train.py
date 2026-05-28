"""Tests for training utilities."""

from src.train import collect_graph_paths


def test_collect_graph_paths_ignores_pt_directories(tmp_path):
    app_dir = tmp_path / "train" / "contextual" / "com.Example.pt"
    app_dir.mkdir(parents=True)
    graph_path = app_dir / "123.pt"
    graph_path.write_bytes(b"not a real graph")

    pattern = str(tmp_path / "train" / "contextual" / "**" / "*.pt")

    assert collect_graph_paths(pattern) == [str(graph_path)]
