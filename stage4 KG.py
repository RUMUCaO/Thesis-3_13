from kg_gen import KGGen

# Initialize KGGen with optional configuration
kg = KGGen(
  model="deepseek/deepseek-v4-pro",  # Default model
  temperature=0.0,        # Default temperature
  api_key="sk-ced541df54664f858fa1ff0f1b916422",  # Optional if set in environment or using a local model
   api_base="https://api.deepseek.com",
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
  input_data=text_input,
  context="Family relationships"
)
# Output: 
# entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# edges={'is brother of', 'is father of', 'is mother of'} 
# relations={('Ben', 'is brother of', 'Josh'), 
#           ('Andrew', 'is father of', 'Josh'), 
#           ('Linda', 'is mother of', 'Josh')}

KGGen.visualize(graph_1, "output.html", open_in_browser=True)