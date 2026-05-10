# Movie Structure Pipeline (Stage1/2/3)

This repository contains a small, modular pipeline for turning long-form movie input
into a constraint-induced structured representation. The pipeline is split into three
independent stages (scripts):

- `stage1_temporal_alignment.py` — Use subtitles (SRT) to build temporal bins (Zt). Optionally extracts visual/audio/text embeddings.
- `stage2_event_segmentation.py` — Aggregate Zt bins into events (E) using change point detection.
- `stage3_structured_induction.py` — Build a sparse event graph S* from events using a simple compatibility function.

Quickstart

1. Install minimal dependencies (recommended in a venv):

```bash
pip install -r requirements.txt
```

2. Run Stage 1 (temporal alignment). If you have `movie.mp4` and `subs.srt`:

```bash
python3 stage1_temporal_alignment.py --video movie.mp4 --srt subs.srt --out zt_bins.json --sample_fps 1.0 --device cpu
```

3. Run Stage 2 (event segmentation):

```bash
python3 stage2_event_segmentation.py --zt zt_bins.json --out events.json
```

4. Run Stage 3 (structured induction):

```bash
python3 stage3_structured_induction.py --events events.json --out structure.json --top_k 3
```

Notes
- The scripts are intentionally lightweight and have optional dependencies. If some libraries
  are not installed (e.g., `transformers`, `librosa`), the pipeline still runs but some
  embeddings will be missing.
- For a production-quality pipeline you should provide GPU for the CLIP model, and add
  face/ReID modules for entity consistency.

Testing

Run pytest to execute the small test suite:

```bash
pytest -q
```
