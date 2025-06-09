import json
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        print(f"Error scraping {url}: {e}")
        return None

def main():
    urls = load_config()
    raw_texts = []
    sources = []

    for url in urls:
        print(f"Scraping {url} ...")
        content = scrape_website(url)
        if content:
            raw_texts.append(content)
            sources.append(url)

    print(f"Scraped {len(raw_texts)} documents.")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Ensure docs is initialized as an empty list
    docs = []  # Initialize docs here

    # Split texts into chunks and build docs list
    for text, src in zip(raw_texts, sources):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            docs.append({
                "text": chunk,
                "metadata": {"source": src}
            })

    print(f"Total chunks created: {len(docs)}")

    # Ensure docs is not empty before proceeding
    if not docs:
        print("No documents to process. Exiting.")
        return

    texts = [doc["text"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]

    print(f"Generating embeddings for {len(texts)} chunks...")

    # Load SentenceTransformer model and embed chunks
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings_data = {
        "embeddings": embeddings,
        "texts": texts,
        "metadatas": metadatas,
    }

    # Save embeddings to a pickle file
    with open('embeddings_fes.pkl', 'wb') as f:
        pickle.dump(embeddings_data, f)

    print("Embeddings saved to embeddings_fes.pkl")

if __name__ == "__main__":
    main()
