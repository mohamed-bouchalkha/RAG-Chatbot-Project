import os
import pickle
import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

# Path where embeddings are stored
embeddings_dir = r"C:\Users\ACER NITRO 5\Documents\GitHub\RAG-Ro-Project\RAG-Chatbot-Project\FesDataScraping"

# Groq setup
try:
    groq_api_key = "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"
    if not groq_api_key:
        groq_api_key = input("Veuillez entrer votre clé API Groq: ")
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.1,
        max_tokens=1000,
    )
    print("Modèle Groq initialisé avec succès!")
except Exception as e:
    print(f"Erreur lors de l'initialisation de Groq: {str(e)}")
    print("Utilisation d'une approche simplifiée...")
    class SimpleLLM:
        def __call__(self, prompt):
            context_start = prompt.find("CONTEXTE:") + 10
            context_end = prompt.find("Basé sur le CONTEXTE")
            context = prompt[context_start:context_end].strip()
            return f"Voici les informations disponibles sur ce sujet :\n\n{context}"
    llm = SimpleLLM()

# Load embedding model
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

print("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
except Exception as e:
    print(f"Error loading embedding model: {str(e)}")
    # Fallback to TF-IDF
    try:
        with open(os.path.join(embeddings_dir, "tfidf_embeddings.pkl"), "rb") as f:
            tfidf_data = pickle.load(f)
        
        vectorizer = tfidf_data["vectorizer"]
        embeddings_list = tfidf_data["embeddings"]
        texts = tfidf_data["texts"]
        metadatas = tfidf_data["metadatas"]
        
        class TfidfEmbedder:
            def __init__(self, vectorizer):
                self.vectorizer = vectorizer
            
            def embed_query(self, query):
                return self.vectorizer.transform([query]).toarray()[0]
        
        embedding_model = TfidfEmbedder(vectorizer)
    except:
        print("Failed to load TF-IDF embedder.")
        exit(1)

# Load main embeddings
try:
    with open(os.path.join(embeddings_dir, "embeddings_data.pkl"), "rb") as f:
        data_main = pickle.load(f)
    embeddings_main = data_main["embeddings"]
    texts_main = data_main["texts"]
    metadatas_main = data_main["metadatas"]
    print("Main embeddings loaded successfully!")
except Exception as e:
    print(f"Error loading main embeddings: {str(e)}")
    exit(1)

# Load Fes embeddings
try:
    with open(os.path.join(embeddings_dir, "embeddings_fes.pkl"), "rb") as f:
        data_fes = pickle.load(f)
    embeddings_fes = data_fes["embeddings"]
    texts_fes = data_fes["texts"]
    metadatas_fes = data_fes["metadatas"]
    print("Fes embeddings loaded successfully!")
except Exception as e:
    print(f"Error loading Fes embeddings: {str(e)}")
    exit(1)

# Check embedding dimensions
dim_main = np.array(embeddings_main).shape[1]
dim_fes = np.array(embeddings_fes).shape[1]
print(f"Main embeddings dimension: {dim_main}")
print(f"Fes embeddings dimension: {dim_fes}")

if dim_main != dim_fes:
    print("Embedding dimension mismatch detected! Recomputing Fes embeddings...")
    embeddings_fes = embedding_model.embed_documents(texts_fes)
    print("Fes embeddings recomputed successfully.")

# Combine embeddings
embeddings_list = np.vstack([np.array(embeddings_main), np.array(embeddings_fes)])
texts = texts_main + texts_fes
metadatas = metadatas_main + metadatas_fes
print(f"Total documents combined: {len(texts)}")

# Define custom retriever compatible with LangChain
class CustomRetriever(BaseRetriever):
    embeddings: np.ndarray
    texts: List[str]
    metadatas: List[dict]
    embedding_model: any
    
    def __init__(self, embeddings, texts, metadatas, embedding_model):
        super().__init__()
        self.embeddings = np.array(embeddings)
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_model = embedding_model
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List:
        if hasattr(self.embedding_model, 'embed_query'):
            query_embedding = self.embedding_model.embed_query(query)
        else:
            query_embedding = self.embedding_model.embed_documents([query])[0]
        
        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / np.maximum(emb_norms, 1e-10)
        
        similarities = np.dot(normalized_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-3:][::-1]
        
        from langchain.schema import Document
        return [
            Document(
                page_content=self.texts[i],
                metadata=self.metadatas[i]
            )
            for i in top_k_indices
        ]

# Create retriever with compression
base_retriever = CustomRetriever(embeddings_list, texts, metadatas, embedding_model)
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Prompt template
template = """
Tu es un assistant intelligent spécialisé dans les informations académiques.
Tu as accès à des documents provenant de différentes formations universitaires.
Tu es aussi un guide de la ville de Fès, au Maroc.

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

# Query processing function
def process_query(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = PROMPT.format(question=query, context=context)
    
    if isinstance(llm, ChatGroq):
        system_message = SystemMessage(content="Tu es un assistant académique qui aide les étudiants avec des informations sur les formations universitaires, et un guide de la ville de Fès, au Maroc.")
        human_message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([system_message, human_message])
        answer = response.content
    else:
        answer = llm(formatted_prompt)
    
    sources = [doc.metadata.get("source", "Source inconnue") for doc in docs]
    sources_str = "Sources: " + ", ".join(sources)
    return answer + "\n\n" + sources_str

# Simple CLI chatbot interface
def chatbot():
    print("Bienvenue dans le chatbot RAG académique avec Groq! Posez vos questions sur les formations universitaires ou la ville de Fès.")
    print("(Tapez 'exit' pour quitter)")
    if isinstance(llm, ChatGroq):
        print(f"Modèle utilisé: Groq - {llm.model_name}")
    
    while True:
        query = input("\nVotre question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        try:
            answer = process_query(query)
            print("\nRéponse:")
            print(answer)
        except Exception as e:
            print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    chatbot()