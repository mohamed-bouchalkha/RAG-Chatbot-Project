import pickle
with open(r"C:\Users\ACER NITRO 5\Documents\GitHub\RAG-Ro-Project\RAG-Chatbot-Project\DataChatbot\embeddings\embeddingsTowTest\embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)

for text in data["texts"]:
    if "MLAIM" in text:
        print(text)
        
        