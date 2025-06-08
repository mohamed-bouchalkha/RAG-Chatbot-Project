import json
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pickle
import os

def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config.get('urls', [])
    except Exception as e:
        print(f"Error loading config: {e}")
        return []

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text().strip() for p in paragraphs])
        return text
    except requests.RequestException as e:
        return f"Error scraping {url}: {e}"

def main():
    urls = load_config()
    texts = []
    sources = []

    for url in urls:
        print(f"Scraping {url}...")
        content = scrape_website(url)
        if content:
            texts.append(content)
            sources.append(url)

    # Split into chunks (optional if long)
    documents = []
    for text, source in zip(texts, sources):
        chunks = text.split('\n')
        for chunk in chunks:
            if chunk.strip():
                documents.append({"text": chunk.strip(), "source": source})

    print(f"Generating embeddings for {len(documents)} chunks...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc["text"] for doc in documents], show_progress_bar=True)

    # Save both the embeddings and texts
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump({
            "documents": documents,
            "embeddings": embeddings
        }, f)

    print("Embeddings saved to embeddings.pkl")

if __name__ == "__main__":
    main()
