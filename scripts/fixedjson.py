import json


with open("data/embeddings/ppc_embeddings.json", "r") as f:
    embeddings_data = json.load(f)

with open("data/processed/ppc_section_original.json", "r") as f:  
    original_texts = json.load(f)  


converted_data = []
for section_num, embedding in embeddings_data.items():
    
    text_content = original_texts.get(section_num, "")
    converted_data.append({
        "section_num": section_num,
        "text": text_content,  
        "embedding": embedding
    })


with open("data/embeddings/ppc_embeddings_fixed.json", "w") as f:
    json.dump(converted_data, f, indent=4)

print("Fixed JSON format saved with actual text content.")