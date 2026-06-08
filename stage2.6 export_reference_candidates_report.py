#!/usr/bin/env python3
"""Export candidate face clusters to an HTML report and CSV.

The report includes a sortable-style table with key fields and representative frames.
If a candidate has no representative_frame path yet, and --video is provided, the script
extracts the midpoint frame into the report assets folder.

Usage:
  python export_reference_candidates_report.py --candidates candidates.json --video 10THAU.mp4 --out report/reference_candidates.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def load_candidates(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("candidates JSON must be a list")
    return data


def load_script_characters(path: str | None) -> list[str]:
    if not path:
        return []
    script_path = Path(path)
    if not script_path.exists():
        return []
    with script_path.open("r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        characters = data.get("characters", [])
    elif isinstance(data, list):
        characters = data
    else:
        return []
    if not isinstance(characters, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in characters:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return cleaned


def extract_frame(video_path: str, time_s: float, out_path: str) -> bool:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_s:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        out_path,
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_frame(candidate: dict[str, Any], video_path: str | None, assets_dir: Path) -> str | None:
    rep = candidate.get("representative_frame")
    if rep:
        rep_path = Path(rep)
        if rep_path.exists():
            return rep_path.as_posix()

    if not video_path:
        return None

    cluster = candidate["cluster"]
    start = float(candidate["start"])
    end = float(candidate["end"])
    time_s = (start + end) / 2.0
    out_path = assets_dir / f"cluster_{cluster}.jpg"
    if extract_frame(video_path, time_s, str(out_path)):
        return out_path.as_posix()
    return None


def resolve_video_path(video: str | None, candidates_path: str) -> str | None:
    if video:
        candidate = Path(video)
        if candidate.exists():
            return candidate.as_posix()
        return video

    candidates_file = Path(candidates_path)
    guesses = [
        candidates_file.with_suffix(".mp4"),
        Path("output_0.5x.mp4"),
        Path("10THAU.mp4"),
    ]
    for guess in guesses:
        if guess.exists():
            return guess.as_posix()

    mp4_files = sorted(Path(".").glob("*.mp4"))
    if len(mp4_files) == 1:
        return mp4_files[0].as_posix()
    return None


def short_json(value: Any, max_len: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def render_report(candidates: list[dict[str, Any]], title: str, script_characters: list[str]) -> str:
    candidates_json = json.dumps(candidates, ensure_ascii=False).replace("</", "<\\/")
    script_characters_json = json.dumps(script_characters, ensure_ascii=False).replace("</", "<\\/")

    rows = []
    for idx, c in enumerate(candidates, start=1):
        speaker_overlaps = c.get("speaker_overlaps", {})
        frame_path = c.get("resolved_frame_path")
        character_id = html.escape(str(c.get("character_id") or ""))
        frame_html = "<div class='no-frame'>no frame</div>"
        if frame_path:
            rel = Path(frame_path).as_posix()
            frame_html = "<a href='{href}' target='_blank'><img src='{href}' alt='cluster {cluster}'></a>".format(
                href=html.escape(rel),
                cluster=c["cluster"],
            )

        row = """
            <tr class='{row_class}'>
              <td><input type='checkbox' class='pick-box' data-cluster='{cluster}' {checked}></td>
              <td><input type='text' class='char-input' data-cluster='{cluster}' value='{character_id}' placeholder='KAT'></td>
              <td>{idx}</td>
              <td><strong>{cluster}</strong></td>
              <td>{start:.3f}</td>
              <td>{end:.3f}</td>
              <td>{duration:.3f}</td>
              <td>{num_samples}</td>
              <td>{active_speakers}</td>
              <td>{dominant_share:.3f}</td>
              <td>{purity:.3f}</td>
              <td>{rank_score:.3f}</td>
              <td>{overlap_count}</td>
              <td><pre>{speaker_overlaps}</pre></td>
              <td>{frame_html}</td>
            </tr>
        """.format(
            row_class="pass" if c.get("passes_thresholds") else "fail",
            checked="checked" if c.get("passes_thresholds") else "",
            cluster=c["cluster"],
            character_id=character_id,
            idx=idx,
            start=c["start"],
            end=c["end"],
            duration=c["duration"],
            num_samples=c["num_samples"],
            active_speakers=c["active_speakers"],
            dominant_share=c.get("dominant_share", 0.0),
            purity=c.get("purity", 0.0),
            rank_score=c.get("rank_score", 0.0),
            overlap_count=c.get("overlap_count", 0),
            speaker_overlaps=html.escape(short_json(speaker_overlaps, 220)),
            frame_html=frame_html,
        )
        rows.append(row)

    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #0b1220;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --line: #243244;
      --pass: #0f5132;
      --fail: #3f1d1d;
      --accent: #60a5fa;
    }}
    body {{ margin: 0; background: linear-gradient(180deg, #0b1020, #111827 55%, #0f172a); color: var(--text); font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta {{ color: var(--muted); margin-bottom: 18px; }}
    .legend {{ display: flex; gap: 16px; align-items: center; margin-bottom: 12px; color: var(--muted); flex-wrap: wrap; }}
    .badge {{ padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line); background: var(--panel); }}
    .character-bank {{ margin: 14px 0 18px; padding: 14px; border: 1px solid var(--line); background: rgba(17,24,39,.75); border-radius: 14px; }}
    .character-bank h2 {{ margin: 0 0 10px; font-size: 16px; color: #dbeafe; }}
    .bank-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .bank-chip {{ border: 1px solid #35506f; background: linear-gradient(180deg, #17304a, #0f172a); color: #e2e8f0; border-radius: 999px; padding: 6px 10px; font-size: 12px; cursor: grab; user-select: none; }}
    .bank-chip:active {{ cursor: grabbing; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0 14px; }}
    button {{ appearance: none; border: 1px solid var(--line); background: #172036; color: var(--text); padding: 10px 14px; border-radius: 10px; cursor: pointer; font-size: 14px; }}
    button:hover {{ border-color: var(--accent); }}
    button.primary {{ background: linear-gradient(180deg, #2563eb, #1d4ed8); border-color: #1d4ed8; }}
    button.ghost {{ background: #0b1220; }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(15, 23, 42, 0.88); border: 1px solid var(--line); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #0f172a; z-index: 1; text-align: left; color: #cbd5e1; }}
    tr.pass {{ background: rgba(16, 185, 129, 0.08); }}
    tr.fail {{ background: rgba(239, 68, 68, 0.05); }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; max-width: 340px; }}
    img {{ max-width: 220px; height: auto; display: block; border-radius: 10px; border: 1px solid var(--line); box-shadow: 0 2px 12px rgba(0,0,0,.25); }}
    .no-frame {{ color: var(--muted); font-style: italic; }}
    .sticky-note {{ margin: 14px 0 18px; padding: 12px 14px; border: 1px solid var(--line); background: rgba(17,24,39,.75); border-radius: 12px; color: var(--muted); }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>{html.escape(title)}</h1>
    <div class='meta'>Sorted by `rank_score` descending. Green rows passed the reference thresholds.</div>
    <div class='legend'>
      <span class='badge'>dominant_share = main speaker / duration</span>
      <span class='badge'>purity = main speaker / total overlap</span>
      <span class='badge'>speaker_overlaps = diarization overlap by speaker</span>
    </div>
    <div class='toolbar'>
      <button class='primary' onclick='selectRecommended()'>Select recommended rows</button>
      <button class='ghost' onclick='clearSelections()'>Clear selection</button>
      <button class='primary' onclick='autoMatchSelected()'>Auto match selected</button>
      <button class='primary' onclick='downloadSelection()'>Export selected clusters JSON</button>
    </div>
    <div class='sticky-note'>Tip: start with the green rows. For multi-speaker clusters, put multiple names in <code>character_id</code> separated by commas (for example: <code>PATRICK, ELLIE</code>). The first name is exported as the primary <code>character_id</code>.</div>
    <div class='character-bank'>
      <h2>Parsed Characters</h2>
      <div class='bank-list' id='character-bank'></div>
    </div>
    <datalist id='character-list'></datalist>
    <table>
      <thead>
        <tr>
          <th>pick</th>
          <th>character_id</th>
          <th>#</th>
          <th>cluster</th>
          <th>start</th>
          <th>end</th>
          <th>duration</th>
          <th>samples</th>
          <th>speakers</th>
          <th>dominant_share</th>
          <th>purity</th>
          <th>rank_score</th>
          <th>overlap_count</th>
          <th>speaker_overlaps</th>
          <th>frame</th>
        </tr>
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </div>
  <script>
    const CANDIDATES = {candidates_json};
    const SCRIPT_CHARACTERS = {script_characters_json};
    const AUTO_SUGGESTIONS = new Map();

    function setCharacterValue(input, value, source='manual') {{
      if (!input) return;
      input.dataset.pendingSetSource = source;
      input.value = value;
      input.focus();
      input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}

    function renderCharacterBank() {{
      const bank = document.getElementById('character-bank');
      const list = document.getElementById('character-list');
      if (!bank) return;
      bank.innerHTML = '';
      if (list) list.innerHTML = '';
      SCRIPT_CHARACTERS.forEach(name => {{
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'bank-chip';
        chip.textContent = name;
        chip.draggable = true;
        chip.addEventListener('dragstart', event => {{
          event.dataTransfer.setData('text/plain', name);
          event.dataTransfer.effectAllowed = 'copy';
        }});
        chip.addEventListener('click', () => {{
          const active = document.activeElement;
          if (active && active.classList && active.classList.contains('char-input')) {{
            setCharacterValue(active, name);
          }}
        }});
        bank.appendChild(chip);

        if (list) {{
          const option = document.createElement('option');
          option.value = name;
          list.appendChild(option);
        }}
      }});
    }}

    function wireCharacterInputs() {{
      document.querySelectorAll('.char-input').forEach(input => {{
        input.setAttribute('list', 'character-list');
        input.dataset.userEdited = input.value.trim() ? '1' : '0';
        input.addEventListener('input', () => {{
          const source = input.dataset.pendingSetSource || '';
          if (source === 'auto') {{
            if (input.dataset.userEdited !== '1') input.dataset.userEdited = '0';
          }} else {{
            input.dataset.userEdited = '1';
          }}
          delete input.dataset.pendingSetSource;
        }});
        input.addEventListener('dragover', event => event.preventDefault());
        input.addEventListener('drop', event => {{
          event.preventDefault();
          const name = event.dataTransfer.getData('text/plain');
          if (name) setCharacterValue(input, name);
        }});
      }});
    }}

    function speakerPeakShare(item) {{
      const overlaps = item.speaker_overlaps || {{}};
      const values = Object.values(overlaps).map(value => Number(value) || 0).filter(value => value > 0);
      if (!values.length) return 0;
      const total = values.reduce((sum, value) => sum + value, 0);
      if (total <= 0) return 0;
      return Math.max(...values) / total;
    }}

    function matchQuality(item) {{
      const rankScore = Math.log1p(Math.max(0, Number(item.rank_score) || 0));
      const purity = Number(item.purity) || 0;
      const dominantShare = Number(item.dominant_share) || 0;
      const samples = Math.log1p(Math.max(0, Number(item.num_samples) || 0));
      const activeSpeakers = Number(item.active_speakers) || 0;
      const overlapCount = Number(item.overlap_count) || 0;
      const peakShare = speakerPeakShare(item);
      const thresholdBonus = item.passes_thresholds ? 1.0 : -1.5;

      return (
        rankScore * 2.0 +
        purity * 3.0 +
        dominantShare * 2.0 +
        peakShare * 2.5 +
        samples * 0.35 -
        activeSpeakers * 0.6 -
        overlapCount * 0.12 +
        thresholdBonus
      );
    }}

    function parseCharacterIds(rawValue) {{
      if (!rawValue) return [];
      return String(rawValue)
        .split(/[;,|]/)
        .map(x => x.trim())
        .filter(Boolean);
    }}

    function autoMatchSelected() {{
      const selectedItems = Array.from(document.querySelectorAll('.pick-box:checked'))
        .map(box => {{
          const row = box.closest('tr');
          if (!row) return null;
          const cluster = Number(box.dataset.cluster);
          const item = CANDIDATES.find(entry => Number(entry.cluster) === cluster);
          const input = row.querySelector('.char-input');
          return {{ row, cluster, item, input }};
        }})
        .filter(entry => entry && entry.item && entry.input);

      if (!selectedItems.length) {{
        alert('No rows selected. Pick one or more clusters first.');
        return;
      }}

      const emptyTargets = selectedItems
        .filter(entry => !entry.input.value.trim())
        .sort((a, b) => matchQuality(b.item) - matchQuality(a.item));

      const names = SCRIPT_CHARACTERS.slice();

      if (!names.length) {{
        alert('No parsed script characters available. Load script_structured.json first.');
        return;
      }}

      let cursor = 0;
      emptyTargets.forEach((entry) => {{
        const speakerCount = Math.max(1, Math.min(3, Number(entry.item.active_speakers) || 1));
        const picked = [];
        for (let i = 0; i < speakerCount && cursor < names.length; i += 1) {{
          picked.push(names[cursor]);
          cursor += 1;
        }}
        if (picked.length) {{
          AUTO_SUGGESTIONS.set(Number(entry.cluster), picked.slice());
          setCharacterValue(entry.input, picked.join(', '), 'auto');
        }}
      }});

      alert('Auto match filled suggested labels for selected rows. You can drag or type to manually confirm/override before export.');
    }}

    renderCharacterBank();
    wireCharacterInputs();

    function selectRecommended() {{
      document.querySelectorAll('.pick-box').forEach(box => {{
        const row = box.closest('tr');
        const isRecommended = row && row.classList.contains('pass');
        box.checked = isRecommended;
      }});
    }}

    function clearSelections() {{
      document.querySelectorAll('.pick-box').forEach(box => box.checked = false);
    }}

    function buildSelection() {{
      const checked = new Set(Array.from(document.querySelectorAll('.pick-box:checked')).map(box => Number(box.dataset.cluster)));
      const inputMap = new Map(Array.from(document.querySelectorAll('.char-input')).map(input => [Number(input.dataset.cluster), input]));
      return CANDIDATES.filter(item => checked.has(Number(item.cluster))).map(item => {{
        const cluster = Number(item.cluster);
        const input = inputMap.get(cluster);
        const rawLabel = input ? input.value.trim() : '';
        const labels = parseCharacterIds(rawLabel);
        const suggestedLabels = AUTO_SUGGESTIONS.get(cluster) || [];
        const userEdited = input ? input.dataset.userEdited === '1' : false;
        const source = labels.length
          ? (userEdited ? 'manual_confirmed' : 'auto_suggested')
          : (suggestedLabels.length ? 'auto_suggested' : 'unassigned');
        return {{
          cluster,
          character_id: labels.length ? labels[0] : null,
          character_ids: labels,
          suggested_character_id: suggestedLabels.length ? suggestedLabels[0] : null,
          suggested_character_ids: suggestedLabels,
          character_id_source: source,
          start: item.start,
          end: item.end,
          duration: item.duration,
          num_samples: item.num_samples,
          dominant_share: item.dominant_share,
          purity: item.purity,
          rank_score: item.rank_score,
          passes_thresholds: item.passes_thresholds,
          speaker_overlaps: item.speaker_overlaps,
          representative_frame: item.resolved_frame_path || item.representative_frame || null,
        }};
      }});
    }}

    function downloadSelection() {{
      const selection = buildSelection();
      const blob = new Blob([JSON.stringify(selection, null, 2)], {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'selected_reference_clusters.json';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.json", help="path to candidates.json")
    parser.add_argument("--video", default=None, help="optional video path to extract missing frames")
    parser.add_argument("--script", default="script_structured.json", help="optional script JSON for draggable character names")
    parser.add_argument("--out", default="report/reference_candidates.html", help="output HTML report")
    parser.add_argument("--csv", default="report/reference_candidates.csv", help="output CSV report")
    parser.add_argument("--title", default="Reference Candidate Clusters", help="report title")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    script_characters = load_script_characters(args.script)
    candidates = sorted(candidates, key=lambda c: c.get("rank_score", 0.0), reverse=True)
    resolved_video = resolve_video_path(args.video, args.candidates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = out_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        frame = ensure_frame(c, resolved_video, assets_dir)
        if frame:
            c["resolved_frame_path"] = os.path.relpath(frame, out_path.parent).replace("\\", "/")
        else:
            c["resolved_frame_path"] = None

    # CSV export
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster",
            "character_id",
            "character_ids",
            "start",
            "end",
            "duration",
            "num_samples",
            "active_speakers",
            "dominant_share",
            "purity",
            "rank_score",
            "passes_thresholds",
            "overlap_count",
            "speaker_overlaps",
            "frame_path",
        ])
        for c in candidates:
            writer.writerow([
                c.get("cluster"),
                c.get("character_id"),
                json.dumps(c.get("character_ids", []), ensure_ascii=False),
                c.get("start"),
                c.get("end"),
                c.get("duration"),
                c.get("num_samples"),
                c.get("active_speakers"),
                c.get("dominant_share"),
                c.get("purity"),
                c.get("rank_score"),
                c.get("passes_thresholds"),
                c.get("overlap_count"),
                json.dumps(c.get("speaker_overlaps", {}), ensure_ascii=False),
                c.get("resolved_frame_path"),
            ])

    html_text = render_report(candidates, args.title, script_characters)
    out_path.write_text(html_text, encoding="utf8")
    print(f"Wrote HTML report to {out_path}")
    print(f"Wrote CSV report to {csv_path}")
    if resolved_video:
        print(f"Frame assets stored in {assets_dir}")
    else:
      print("Warning: no video path found, so clusters without existing representative_frame will show 'no frame'.")


if __name__ == "__main__":
    main()
