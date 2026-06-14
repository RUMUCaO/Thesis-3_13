import json

from kg_gen import KGGen

# Initialize KGGen with optional configuration
def extract_full_text(json_path: str) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    full_text_parts = [scene.get("text", "") for scene in scenes]
    return "\n\n".join(full_text_parts)  # 用两个换行分隔每个场景

# 2. 提取文本
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

# 3. 初始化 KGGen
kg = KGGen(
    model="deepseek/deepseek-v4-flash",   # 或者 "deepseek-chat"（DeepSeek API 实际模型名）
    temperature=0.0,
    api_key="sk-3d5831b61f424375aa16dff723d3ec96",   # 替换成你自己的 key
    api_base="https://api.deepseek.com",
)

# 4. 生成知识图谱
graph = kg.generate(
    input_data=full_text,
    context=context_instruction,
)

# 5. 可视化
KGGen.visualize(graph, "script_relationships.html", open_in_browser=True)