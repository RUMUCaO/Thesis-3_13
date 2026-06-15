import json

from kg_gen import KGGen

# Initialize KGGen with optional configuration
def extract_full_text(json_path: str) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    full_text_parts = [scene.get("text", "") for scene in scenes]
    return "\n\n".join(full_text_parts)  # Separate each scene with two newlines.

# 2. Extract the text
full_text = extract_full_text("generated_scripts.json")
print(f"Extracted {len(full_text)} characters")

allowed_characters = [
    "BIANCA", "BLAISE", "BOGEY", "BRUCE", "CAMERON", "CHAPIN", "CHASTITY",
    "CLEM", "DEREK", "JOEY", "KAT", "MANDELIA", "MANDELLA", "MICHAEL",
    "NEARBY", "PATRICK", "PEPE", "PERKY", "SCURVY", "SHARON", "SKIPPY",
    "TREVOR", "WALTER"
]

context_instruction = (
    f"You are extracting entities from a movie script. "
    f"Only the following character names may be extracted as PERSON entities: "
    f"{', '.join(sorted(allowed_characters))}. "
    f"Do not invent any other names. If a name is not in this list, ignore it."
)

# 3. Initialize KGGen
kg = KGGen(
    model="deepseek/deepseek-v4-flash",   # or "deepseek-chat"
    temperature=0.0,
    api_key="sk-3d5831b61f424375aa16dff723d3ec96",   # your own DeepSeek API key
    api_base="https://api.deepseek.com",
)

# 4. generate the knowledge graph
graph = kg.generate(
    input_data=full_text,
    context=context_instruction,
)

# 5. Visualization
KGGen.visualize(graph, "script_relationships.html", open_in_browser=True)