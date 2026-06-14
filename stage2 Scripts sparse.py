import pymupdf
import json
import importlib
import os
import re
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

# ----------------------------
# CONFIG
# ----------------------------
PDF_PATH = "01_10_Things_I_Hate_About_You.pdf"
OUTPUT_JSON = "script_structured.json"
DEFAULT_LLM_MODEL = os.environ.get("SCRIPT_PARSER_LLM_MODEL", "deepseek-v4-pro")
LLM_CHUNK_CHAR_LIMIT = int(os.environ.get("SCRIPT_PARSER_LLM_CHUNK_CHAR_LIMIT", "2800"))
LLM_BASE_URL = os.environ.get("SCRIPT_PARSER_LLM_BASE_URL", "https://api.deepseek.com")
LLM_REQUEST_TIMEOUT_S = float(os.environ.get("SCRIPT_PARSER_LLM_TIMEOUT_S", "60"))
LLM_MAX_RETRIES = int(os.environ.get("SCRIPT_PARSER_LLM_MAX_RETRIES", "0"))
RUN_PDF_PATH = os.environ.get("SCRIPT_PARSER_PDF_PATH", PDF_PATH)
RUN_OUTPUT_JSON = os.environ.get("SCRIPT_PARSER_OUTPUT_JSON", OUTPUT_JSON)
RUN_USE_LLM = os.environ.get("SCRIPT_PARSER_USE_LLM", "0") == "1"
RUN_NORMALIZE_CHARACTERS = os.environ.get("SCRIPT_PARSER_NORMALIZE_CHARACTERS", "0") == "1"
RUN_LLM_MODE = os.environ.get("SCRIPT_PARSER_LLM_MODE", "extract")
RUN_LLM_MODEL = os.environ.get("SCRIPT_PARSER_LLM_MODEL", DEFAULT_LLM_MODEL)
RUN_LLM_BASE_URL = os.environ.get("SCRIPT_PARSER_LLM_BASE_URL", LLM_BASE_URL)
SCENE_OUTPUT_KEYS = (
    "scene_id",
    "heading",
    "location",
    "time_of_day",
    "characters",
    "dialogue_blocks",
    "action_blocks",
)

# ----------------------------
# LOAD PDF TEXT
# ----------------------------
def load_pdf_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)

    return "\n".join(pages)


# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text):
    lines = []
    for raw_line in text.splitlines():
        normalized = re.sub(r"[ \t]+", " ", raw_line).strip()
        lines.append(normalized)

    page_pattern = re.compile(
        r"^(?:"
        r"Page\s*\d+(?:\s+of\s+\d+)?\s*|"   # Page 1, Page 1 of 130
        r"\d+\s*\.?\s*|"                     # 1. or 2 
        r"\d+\s+of\s+\d+\s*|"                # 1 of 10
        r"\[\d+\]\s*"                        # [1]
        r")$",
        flags=re.IGNORECASE
    )
    filtered_lines = [line for line in lines if not page_pattern.match(line)]

    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ----------------------------
# DETECT CHARACTER LINES
# (ALL CAPS heuristic)
# ----------------------------
SCENE_HEADING_PATTERN = re.compile(
    r"^(?P<kind>INT\.?|EXT\.?|INT/EXT\.?|I/E\.?)\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
CHARACTER_PATTERN = re.compile(r"^[A-Z][A-Z\s\(\)\-'.]{1,}$")
PARENTHETICAL_PATTERN = re.compile(r"^\([^)]*\)$")

BAD_CHARACTER_PREFIXES = (
    "INT",
    "EXT",
    "I/E",
    "CUT",
    "FADE",
    "DISSOLVE",
)

CHARACTER_TITLE_PREFIXES = {
    "MR",
    "MRS",
    "MS",
    "MISS",
    "DR",
    "PROF",
    "PROFESSOR",
    "CAPT",
    "CAPTAIN",
    "OFFICER",
    "SGT",
    "SIR",
    "LADY",
}

CHARACTER_PREPOSITION_START = {
    "AT",
    "IN",
    "ON",
    "NEAR",
    "ACROSS",
    "INSIDE",
    "OUTSIDE",
    "FROM",
    "TO",
    "WITH",
}

CHARACTER_PRONOUN_START = {
    "HIS",
    "HER",
    "THEIR",
    "OUR",
    "MY",
    "YOUR",
}

CAMERA_DIRECTION_TOKENS = {
    "POV",
    "ANGLE",
    "SHOT",
    "VIEW",
    "CLOSE",
    "WIDE",
}

GENERIC_ROLE_LABELS = {
    "LEAD",
    "BARTENDER",
    "BOY",
    "GIRL",
    "GUY",
    "KID",
    "JOCK",
    "TEACHER",
    "RIDER",
    "COWBOY",
    "MISCREANT",
    "COHORT",
    "SINGER",
}

AUXILIARY_FRAGMENT_TOKENS = {
    "BE",
    "IS",
    "ARE",
    "WAS",
    "WERE",
    "WILL",
    "SHALL",
    "WOULD",
    "SHOULD",
}

def is_character_line(line):
    line = line.strip()
    if not line:
        return False
    if SCENE_HEADING_PATTERN.match(line):
        return False
    return bool(CHARACTER_PATTERN.match(line)) and len(line.split()) <= 4


def is_valid_character(name):
    n = re.sub(r"\([^)]*\)", "", name).strip().upper()
    if not n:
        return False

    if n.endswith((".", "!", "?")):
        return False

    if any(n.startswith(prefix) for prefix in BAD_CHARACTER_PREFIXES):
        return False

    if " AND " in n or "&" in n:
        return False

    if "-" in n:
        return False

    if re.search(r"\d", n):
        return False

    words = [w for w in re.split(r"\s+", n) if w]
    if len(words) == 0 or len(words) > 3:
        return False

    normalized_words = [re.sub(r"[^A-Z]", "", w) for w in words]
    normalized_words = [w for w in normalized_words if w]
    if len(normalized_words) == 0:
        return False

    first = normalized_words[0]
    if first in CHARACTER_PREPOSITION_START:
        return False
    if first in CHARACTER_PRONOUN_START:
        return False

    if any(w in CAMERA_DIRECTION_TOKENS for w in normalized_words):
        return False

    if all(w in GENERIC_ROLE_LABELS for w in normalized_words):
        return False

    if all(w in AUXILIARY_FRAGMENT_TOKENS for w in normalized_words):
        return False

    # Keep this lexical filter generic: each token should look like a screenplay label token.
    if not all(re.fullmatch(r"[A-Z][A-Z'.]*", w) for w in words):
        return False

    return True


def _has_dialogue_payload(lines, current_index):
    j = current_index + 1
    while j < len(lines):
        candidate = lines[j].strip()
        if not candidate:
            return False
        if SCENE_HEADING_PATTERN.match(candidate):
            return False
        if is_character_line(candidate) and is_valid_character(candidate):
            return False
        if PARENTHETICAL_PATTERN.match(candidate):
            j += 1
            continue
        return True
    return False


def canonicalize_character_name(name):
    n = re.sub(r"\([^)]*\)", "", name).strip().upper()
    if not n:
        return n

    words = [w for w in re.split(r"\s+", n) if w]
    if not words:
        return n

    first_norm = re.sub(r"[^A-Z]", "", words[0])
    if first_norm in CHARACTER_TITLE_PREFIXES and len(words) > 1:
        words = words[1:]

    canonical = " ".join(words)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return canonical


def apply_character_canonicalization(scenes):
    alias_to_canonical = {}

    for scene in scenes:
        normalized_characters = []
        for character in scene.get("characters", []):
            canonical = canonicalize_character_name(character)
            if not canonical:
                continue
            alias_to_canonical[character] = canonical
            _append_unique(normalized_characters, canonical)
        scene["characters"] = normalized_characters

        for block in scene.get("dialogue_blocks", []):
            speaker = block.get("speaker")
            if not isinstance(speaker, str):
                continue
            canonical = canonicalize_character_name(speaker)
            if not canonical:
                continue
            alias_to_canonical[speaker] = canonical
            block["speaker"] = canonical

    return dict(sorted(alias_to_canonical.items(), key=lambda item: item[0]))


def parse_scene_heading(line):
    match = SCENE_HEADING_PATTERN.match(line.strip())
    if not match:
        return {"heading": line.strip(), "location": None, "time_of_day": None}

    rest = match.group("rest").strip()
    location = rest
    time_of_day = None

    if "-" in rest:
        parts = [part.strip() for part in rest.split("-") if part.strip()]
        if parts:
            location = parts[0]
        if len(parts) > 1:
            time_of_day = parts[-1]

    return {
        "heading": line.strip(),
        "location": location or None,
        "time_of_day": time_of_day,
    }


def build_scene(scene_id, heading=None):
    return {
        "scene_id": scene_id,
        "heading": heading["heading"] if heading else None,
        "location": heading["location"] if heading else None,
        "time_of_day": heading["time_of_day"] if heading else None,
        "characters": [],
        "dialogue_blocks": [],
        "action_blocks": [],
        "raw_text_span": [],
    }


def _append_unique(items, value):
    if value not in items:
        items.append(value)


def _maybe_add_action(scene, text):
    cleaned = text.strip()
    if cleaned:
        scene["action_blocks"].append(cleaned)


def _maybe_add_dialogue(scene, speaker, text):
    cleaned = re.sub(r"\([^)]*\)", "", text).strip()
    if cleaned:
        scene["dialogue_blocks"].append({"speaker": speaker, "text": cleaned})


def _llm_runtime_check(base_url=LLM_BASE_URL):
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] Skipped: missing DEEPSEEK_API_KEY (or OPENAI_API_KEY fallback).")
        return False

    try:
        importlib.import_module("openai")
    except Exception as exc:
        print(f"[LLM] Skipped: openai package is not available in this Python environment: {exc}")
        return False

    print(f"[LLM] Enabled: DeepSeek endpoint {base_url}")
    return True


def split_text_into_chunks(text, max_chars=LLM_CHUNK_CHAR_LIMIT):
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_length = 0

    def flush_current():
        nonlocal current_lines, current_length
        if current_lines:
            chunks.append("\n".join(current_lines).strip())
            current_lines = []
            current_length = 0

    for line in lines:
        if len(line) > max_chars:
            flush_current()
            for start in range(0, len(line), max_chars):
                chunks.append(line[start:start + max_chars].strip())
            continue

        line_length = len(line) + 1
        if current_lines and current_length + line_length > max_chars:
            flush_current()

        current_lines.append(line)
        current_length += line_length

    flush_current()

    return [chunk for chunk in chunks if chunk]


def validate_llm_scene_schema(payload):
    if not isinstance(payload, dict):
        return None

    if set(payload.keys()) != set(SCENE_OUTPUT_KEYS):
        return None

    if not isinstance(payload.get("scene_id"), int):
        return None

    for scalar_key in ("heading", "location", "time_of_day"):
        value = payload.get(scalar_key)
        if value is not None and not isinstance(value, str):
            return None

    characters = payload.get("characters")
    if not isinstance(characters, list) or any(not isinstance(item, str) for item in characters):
        return None

    dialogue_blocks = payload.get("dialogue_blocks")
    if not isinstance(dialogue_blocks, list):
        return None
    for block in dialogue_blocks:
        if not isinstance(block, dict):
            return None
        if set(block.keys()) != {"speaker", "text"}:
            return None
        if not isinstance(block.get("speaker"), str) or not isinstance(block.get("text"), str):
            return None

    action_blocks = payload.get("action_blocks")
    if not isinstance(action_blocks, list) or any(not isinstance(item, str) for item in action_blocks):
        return None

    return {
        "scene_id": int(payload["scene_id"]),
        "heading": payload.get("heading"),
        "location": payload.get("location"),
        "time_of_day": payload.get("time_of_day"),
        "characters": characters,
        "dialogue_blocks": dialogue_blocks,
        "action_blocks": action_blocks,
    }


def merge_llm_scene_chunks(scene_id, chunk_payloads):
    merged = {
        "scene_id": scene_id,
        "heading": None,
        "location": None,
        "time_of_day": None,
        "characters": [],
        "dialogue_blocks": [],
        "action_blocks": [],
    }

    for payload in chunk_payloads:
        if payload.get("heading") and merged["heading"] is None:
            merged["heading"] = payload["heading"]
        if payload.get("location") and merged["location"] is None:
            merged["location"] = payload["location"]
        if payload.get("time_of_day") and merged["time_of_day"] is None:
            merged["time_of_day"] = payload["time_of_day"]

        for speaker in payload.get("characters", []):
            _append_unique(merged["characters"], speaker)

        for block in payload.get("dialogue_blocks", []):
            merged["dialogue_blocks"].append(block)

        for action in payload.get("action_blocks", []):
            merged["action_blocks"].append(action)

    return merged


def _parse_scenes_rule_based(text):
    lines = text.split("\n")
    scenes = []
    current_scene = build_scene(0)
    current_speaker = None
    scene_counter = 0

    def flush_scene():
        nonlocal current_scene, scene_counter, current_speaker
        if current_scene["raw_text_span"]:
            scenes.append(current_scene)
        scene_counter += 1
        current_scene = build_scene(scene_counter)
        current_speaker = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_speaker = None
            continue

        if SCENE_HEADING_PATTERN.match(stripped):
            flush_scene()
            heading = parse_scene_heading(stripped)
            current_scene["heading"] = heading["heading"]
            current_scene["location"] = heading["location"]
            current_scene["time_of_day"] = heading["time_of_day"]
            current_scene["raw_text_span"].append(stripped)
            continue

        current_scene["raw_text_span"].append(stripped)

        if is_character_line(stripped):
            if is_valid_character(stripped) and _has_dialogue_payload(lines, i):
                current_speaker = stripped
                _append_unique(current_scene["characters"], current_speaker)
            else:
                current_speaker = None
                _maybe_add_action(current_scene, stripped)
            continue

        if current_speaker and not PARENTHETICAL_PATTERN.match(stripped):
            _maybe_add_dialogue(current_scene, current_speaker, stripped)
            continue

        _maybe_add_action(current_scene, stripped)

    flush_scene()
    return scenes


def _call_llm_for_scene_chunk(
    scene_id,
    scene_text,
    chunk_index,
    chunk_total,
    model_name=DEFAULT_LLM_MODEL,
    base_url=LLM_BASE_URL,
):
    try:
        openai_module = importlib.import_module("openai")
    except Exception as exc:
        print(f"[LLM] Scene {scene_id} chunk {chunk_index + 1}/{chunk_total}: skipped because openai import failed: {exc}")
        return None

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"[LLM] Scene {scene_id} chunk {chunk_index + 1}/{chunk_total}: skipped because no API key is set.")
        return None

    client = openai_module.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=LLM_REQUEST_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )
    prompt = (
        "Extract structured screenplay data from the following screenplay chunk. "
        "Do not summarize, do not infer missing information, do not reorder lines, and do not repair existing structure. "
        "Return JSON only and match the schema exactly: no extra keys, no missing fields.\n\n"
        "Schema:\n"
        "{\n"
        '  "scene_id": 1,\n'
        '  "heading": null,\n'
        '  "location": null,\n'
        '  "time_of_day": null,\n'
        '  "characters": [],\n'
        '  "dialogue_blocks": [{"speaker": "", "text": ""}],\n'
        '  "action_blocks": []\n'
        "}\n\n"
        f"Scene id: {scene_id}\n"
        f"Chunk {chunk_index + 1}/{chunk_total}\n"
        f"Scene text:\n{scene_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You extract screenplay structure only. Output strict JSON with exactly the requested keys."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        print(f"[LLM] Scene {scene_id} chunk {chunk_index + 1}/{chunk_total}: DeepSeek request failed: {exc}")
        return None

    if not response.choices:
        return None

    content = response.choices[0].message.content
    if not content:
        return None

    try:
        return validate_llm_scene_schema(json.loads(content))
    except Exception as exc:
        print(f"[LLM] Scene {scene_id} chunk {chunk_index + 1}/{chunk_total}: invalid JSON/schema from DeepSeek: {exc}")
        return None


def apply_llm_extraction(scenes, model_name=DEFAULT_LLM_MODEL, base_url=LLM_BASE_URL):
    extracted = []
    print(f"[LLM] Extracting {len(scenes)} scenes with timeout={LLM_REQUEST_TIMEOUT_S}s, retries={LLM_MAX_RETRIES}")
    for scene in scenes:
        scene_text = "\n".join(scene.get("raw_text_span", []))
        chunks = split_text_into_chunks(scene_text)
        chunk_payloads = []
        print(f"[LLM] Scene {scene['scene_id']}: {len(chunks)} chunk(s)")

        for chunk_index, chunk_text in enumerate(chunks):
            print(f"[LLM] Scene {scene['scene_id']} chunk {chunk_index + 1}/{len(chunks)}: sending to DeepSeek")
            llm_chunk = _call_llm_for_scene_chunk(
                scene_id=scene["scene_id"],
                scene_text=chunk_text,
                chunk_index=chunk_index,
                chunk_total=len(chunks),
                model_name=model_name,
                base_url=base_url,
            )
            if isinstance(llm_chunk, dict):
                chunk_payloads.append(llm_chunk)
                print(f"[LLM] Scene {scene['scene_id']} chunk {chunk_index + 1}/{len(chunks)}: received valid JSON")
            else:
                print(f"[LLM] Scene {scene['scene_id']} chunk {chunk_index + 1}/{len(chunks)}: no valid JSON returned")

        merged = merge_llm_scene_chunks(scene["scene_id"], chunk_payloads) if chunk_payloads else {
            "scene_id": scene["scene_id"],
            "heading": None,
            "location": None,
            "time_of_day": None,
            "characters": [],
            "dialogue_blocks": [],
            "action_blocks": [],
        }

        enriched_scene = dict(scene)
        enriched_scene["llm_extraction"] = merged
        extracted.append(enriched_scene)

    return extracted


def build_diff_fixes(rule_scene, llm_scene):
    fixes = []
    rule_characters = set(rule_scene.get("characters", []))
    llm_characters = set(llm_scene.get("characters", []))

    for character in sorted(llm_characters - rule_characters):
        fixes.append({"scene_id": rule_scene["scene_id"], "add_character": character})

    if not rule_scene.get("heading") and llm_scene.get("heading"):
        fixes.append({"scene_id": rule_scene["scene_id"], "set_heading": llm_scene["heading"]})

    if not rule_scene.get("location") and llm_scene.get("location"):
        fixes.append({"scene_id": rule_scene["scene_id"], "set_location": llm_scene["location"]})

    if not rule_scene.get("time_of_day") and llm_scene.get("time_of_day"):
        fixes.append({"scene_id": rule_scene["scene_id"], "set_time_of_day": llm_scene["time_of_day"]})

    return fixes


# ----------------------------
# PARSE SCRIPT
# ----------------------------
def parse_script(text, normalize_characters=False):
    characters = set()
    scenes = _parse_scenes_rule_based(text)
    character_alias_map = {}

    if normalize_characters:
        character_alias_map = apply_character_canonicalization(scenes)

    for scene in scenes:
        for speaker in scene.get("characters", []):
            characters.add(speaker)

    result: dict[str, Any] = {"characters": sorted(list(characters)), "scenes": scenes}
    if normalize_characters:
        result["character_alias_map"] = character_alias_map
    return result


# ----------------------------
# MAIN
# ----------------------------
def main():
    raw_text = load_pdf_text(RUN_PDF_PATH)
    cleaned = clean_text(raw_text)

    structured = parse_script(cleaned, normalize_characters=RUN_NORMALIZE_CHARACTERS)

    if RUN_USE_LLM:
        _llm_runtime_check(RUN_LLM_BASE_URL)
        if RUN_LLM_MODE == "extract":
            structured["scenes"] = apply_llm_extraction(
                structured["scenes"],
                model_name=RUN_LLM_MODEL,
                base_url=RUN_LLM_BASE_URL,
            )
        else:  # diff mode
            llm_scenes = apply_llm_extraction(
                structured["scenes"],
                model_name=RUN_LLM_MODEL,
                base_url=RUN_LLM_BASE_URL,
            )
            fixes_by_scene = []
            for rule_scene, llm_scene in zip(structured["scenes"], llm_scenes):
                fixes_by_scene.extend(build_diff_fixes(rule_scene, llm_scene.get("llm_extraction", {})))
            structured["llm_fixes"] = fixes_by_scene

        structured["characters"] = sorted({speaker for scene in structured["scenes"] for speaker in scene.get("characters", [])})

    print("Characters found:", len(structured["characters"]))
    print("Scenes found:", len(structured["scenes"]))

    with open(RUN_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print("Saved →", RUN_OUTPUT_JSON)

if __name__ == "__main__":
    main()