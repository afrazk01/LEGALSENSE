import pickle

texts = pickle.load(open("data/section_texts.pkl", "rb"))

print("✅ Sample section text:")
for i in range(3):
    print(f"{i+1}. {texts[i]}")
