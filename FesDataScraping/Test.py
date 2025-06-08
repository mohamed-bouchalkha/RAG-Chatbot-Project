import pickle
with open("embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)

for text in data["texts"]:
    if "WISD" in text:
        print(text)
        