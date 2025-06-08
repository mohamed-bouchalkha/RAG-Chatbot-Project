import os
import pickle
import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# === Load embeddings from both files ===
def load_embeddings_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    texts = data.get("texts", [])
    metadatas = data.get("metadatas", [])
    embeddings = np.array(data.get("embeddings", []))
    
    documents = []
    for i in range(len(texts)):
        source = metadatas[i].get("source", "unknown") if i < len(metadatas) else "unknown"
        documents.append({"text": texts[i], "source": source})
    return documents, embeddings

def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["documents"], np.array(data["embeddings"])

docs1, emb1 = load_embeddings_data("embeddings_data.pkl")
docs2, emb2 = load_embeddings("embeddings.pkl")

documents = docs1 + docs2
embeddings = np.vstack([emb1, emb2])

print(f"✅ Loaded {len(docs1)} + {len(docs2)} documents = {len(documents)} total")
print(f"📐 Embeddings shape: {embeddings.shape}")

# === Custom Retriever class ===
class CustomRetriever:
    def __init__(self, embeddings, documents, embedding_model):
        self.embeddings = np.array(embeddings)
        self.documents = documents
        self.embedding_model = embedding_model
    
    def get_relevant_documents(self, query, k=5):
        # Clean and encode query
        cleaned_query = self.clean_query(query)
        query_embedding = self.embedding_model.encode([cleaned_query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / np.maximum(emb_norms, 1e-10)
        
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        from langchain.schema import Document
        
        return [
            Document(
                page_content=self.documents[i]["text"],
                metadata={"source": self.documents[i].get("source", "unknown")}
            )
            for i in top_k_indices
        ]

    def clean_query(self, query):
        query = query.lower()
        query = re.sub(r'[^\w\s]', '', query)
        tokens = query.split()
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return " ".join(filtered_tokens)

# === Load SentenceTransformer model for embedding queries ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Create retriever instance ===
retriever = CustomRetriever(embeddings, documents, embedding_model)

# === Define prompt template ===
template = """
Tu es un assistant intelligent spécialisé dans les informations académiques.
Tu as accès à des documents provenant de différentes formations universitaires.

QUESTION: {question}

CONTEXTE:
{context}

Basé sur le CONTEXTE fourni ci-dessus (et pas sur tes connaissances générales), 
réponds à la QUESTION de manière claire, précise et utile.
Si la réponse n'est pas présente dans le CONTEXTE, indique simplement que tu ne sais pas ou que l'information n'est pas disponible.
Indique la source des informations que tu donnes (nom du fichier).
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

# === Initialize Groq model ===
try:
    GROQ_API_KEY = os.getenv("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")
    if not GROQ_API_KEY:
        GROQ_API_KEY = input("Veuillez entrer votre clé API Groq: ")
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=1000,
    )
    print("✅ Modèle Groq initialisé avec succès !")
except Exception as e:
    print(f"⚠️ Erreur lors de l'initialisation de Groq: {e}")
    print("Utilisation d'un fallback simple.")
    
    class SimpleLLM:
        def __call__(self, prompt):
            context_start = prompt.find("CONTEXTE:") + 10
            context_end = prompt.find("Basé sur le CONTEXTE")
            context = prompt[context_start:context_end].strip()
            return f"Voici les informations disponibles :\n\n{context}"
    llm = SimpleLLM()

# === Process query with RAG ===
def process_query(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = PROMPT.format(question=query, context=context)

    if isinstance(llm, ChatGroq):
        system_message = SystemMessage(content="Tu es un assistant académique qui aide les étudiants avec des informations sur les formations universitaires.")
        human_message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([system_message, human_message])
        answer = response.content
    else:
        answer = llm(formatted_prompt)
    
    sources = [doc.metadata.get("source", "Source inconnue") for doc in docs]
    sources_str = "Sources: " + ", ".join(sources)
    return answer + "\n\n" + sources_str

# === Simple chatbot interface ===
def chatbot():
    print("Bienvenue dans le chatbot RAG académique avec Groq! Posez vos questions sur les formations universitaires.")
    print("(Tapez 'exit' pour quitter)")
    if isinstance(llm, ChatGroq):
        print(f"Modèle utilisé: Groq - {llm.model_name}")
    while True:
        query = input("\nVotre question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        try:
            answer = process_query(query)
            print("\n💬 Réponse:")
            print(answer)
        except Exception as e:
            print(f"⚠️ Erreur: {e}")

if __name__ == "__main__":
    chatbot()
