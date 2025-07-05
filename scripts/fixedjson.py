import json


with open("data/embeddings/ppc_embeddings.json", "r") as f:
    original_data = json.load(f)


converted_data = [
    {"text": key, "embedding": value}
    for key, value in original_data.items()
]


with open("data/embeddings/ppc_embeddings_fixed.json", "w") as f:
    json.dump(converted_data, f, indent=4)

print("Fixed JSON format saved.")
