import json
import os
import tempfile

from stage1_temporal_alignment import parse_srt
from stage2_event_segmentation import compute_distance_series, aggregate_bins_to_events
from stage3_structured_induction import build_structure


def test_parse_srt_basic(tmp_path):
    srt = tmp_path / "test.srt"
    srt.write_text("""1
00:00:00,000 --> 00:00:02,000
Hello world.

2
00:00:02,000 --> 00:00:04,000
Another line.
""")
    items = parse_srt(str(srt))
    assert len(items) == 2
    assert items[0][0] == 0.0
    assert items[0][2].startswith("Hello")


def test_segmentation_aggregate():
    # create 4 bins with synthetic vector embeddings
    bins = []
    for i in range(4):
        bins.append({
            "start": float(i * 2),
            "end": float((i + 1) * 2),
            "text": f"t{i}",
            "visual_emb": [1.0 if j == i else 0.0 for j in range(4)],
            "audio_emb": None,
            "text_emb": None,
        })
    d, X = compute_distance_series(bins)
    # there should be 3 distances
    assert d.shape[0] == 3
    # use simple threshold detector
    cuts = [i + 1 for i, v in enumerate(d) if v > 0.5]
    events = aggregate_bins_to_events(bins, cuts)
    assert len(events) >= 1


def test_structure_basic():
    evs = [
        {"start": 0.0, "end": 2.0, "visual_emb": [1, 0, 0]},
        {"start": 2.0, "end": 4.0, "visual_emb": [0, 1, 0]},
        {"start": 4.0, "end": 6.0, "visual_emb": [0, 0, 1]},
    ]
    s = build_structure(evs, top_k=2)
    assert "nodes" in s and "edges" in s
    assert len(s["nodes"]) == 3
    # edges should point to other nodes
    assert all(e["source"] != e["target"] for e in s["edges"])


def test_sce_metrics():
    from evaluate_sce import kendall_tau_ordering, align_score, consensus_score

    evs_a = [
        {"start": 0.0, "text_emb": [1, 0, 0]},
        {"start": 2.0, "text_emb": [0, 1, 0]},
        {"start": 4.0, "text_emb": [0, 0, 1]},
    ]
    evs_b = [
        {"start": 0.1, "text_emb": [1, 0, 0]},
        {"start": 2.1, "text_emb": [0, 1, 0]},
        {"start": 4.1, "text_emb": [0, 0, 1]},
    ]
    tau = kendall_tau_ordering(evs_a, evs_b)
    assert tau > 0.9
    a = align_score(evs_a, evs_b, emb_key="text_emb")
    assert 0.9 <= a <= 1.0
    cons = consensus_score([evs_a, evs_b], emb_key="text_emb")
    assert 0.9 <= cons <= 1.0
