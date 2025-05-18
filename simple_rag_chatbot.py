import os
import pickle
import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq  # Import pour Groq
from langchain_core.messages import HumanMessage, SystemMessage

# Chemin vers le dossier où sont stockés les embeddings
embeddings_dir = r"C:\Users\hp\Desktop\RAG-Chatbot-Project\DataChatbot\embeddings"

# Définir une classe pour le retriever compatible avec LangChain
class CustomRetriever:
    def __init__(self, embeddings, texts, metadatas, embedding_model):
        self.embeddings = np.array(embeddings)
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_model = embedding_model
    
    def get_relevant_documents(self, query, k=3):
        """Recherche les documents les plus similaires à partir d'une requête"""
        # Encoder la requête
        if hasattr(self.embedding_model, 'embed_query'):
            query_embedding = self.embedding_model.embed_query(query)
        else:
            query_embedding = self.embedding_model.embed_documents([query])[0]
        
        query_embedding = np.array(query_embedding)
        
        # Normaliser les embeddings
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Normaliser tous les embeddings de la base
        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / np.maximum(emb_norms, 1e-10)
        
        # Calculer les similarités cosinus
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        # Obtenir les indices des documents les plus similaires
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Convertir en documents LangChain
        from langchain.schema import Document
        
        return [
            Document(
                page_content=self.texts[i],
                metadata=self.metadatas[i]
            )
            for i in top_k_indices
        ]

# Charger les embeddings
try:
    with open(os.path.join(embeddings_dir, "embeddings_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    embeddings_list = data["embeddings"]
    texts = data["texts"]
    metadatas = data["metadatas"]
    
    print("Embeddings chargés avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement des embeddings: {str(e)}")
    exit(1)

# Charger le modèle d'embedding
try:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
except Exception as e:
    print(f"Erreur lors du chargement du modèle d'embedding: {str(e)}")
    
    # Charger l'alternative TF-IDF
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
                """Encoder une requête avec TF-IDF"""
                query_vec = self.vectorizer.transform([query]).toarray()[0]
                return query_vec
        
        embedding_model = TfidfEmbedder(vectorizer)
    except:
        print("Impossible de charger l'embedder TF-IDF.")
        exit(1)

# Créer le retriever personnalisé
retriever = CustomRetriever(embeddings_list, texts, metadatas, embedding_model)

# Template pour le prompt
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

# Configuration de Groq
try:
    # Vérifier si la clé API Groq est disponible
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        # Si la clé n'est pas dans les variables d'environnement, demander à l'utilisateur
        groq_api_key = input("Veuillez entrer votre clé API Groq: ")
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Initialiser le modèle Groq
    # Options de modèles: 'llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b', 'gemma-7b-it'
    llm = ChatGroq(
        model_name="llama3-8b-8192",  # Modèle par défaut, peut être modifié selon vos besoins
        temperature=0.1,
        max_tokens=1000,
    )
    
    print(f"Modèle Groq initialisé avec succès!")
    
except Exception as e:
    print(f"Erreur lors de l'initialisation de Groq: {str(e)}")
    print("Utilisation d'une approche simplifiée...")
    
    # Fallback si Groq n'est pas disponible
    class SimpleLLM:
        def __call__(self, prompt):
            # Extraire la question et le contexte du prompt
            context_start = prompt.find("CONTEXTE:") + 10
            context_end = prompt.find("Basé sur le CONTEXTE")
            context = prompt[context_start:context_end].strip()
            
            # Retourner une réponse simplifiée basée uniquement sur le contexte
            return f"Voici les informations disponibles sur ce sujet :\n\n{context}"
    
    llm = SimpleLLM()

# Fonction pour le RAG avec Groq
def process_query(query):
    # Récupérer les documents pertinents
    docs = retriever.get_relevant_documents(query)
    
    # Extraire le contexte
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Construire le prompt
    formatted_prompt = PROMPT.format(question=query, context=context)
    
    # Pour Groq, utiliser l'interface de chat
    if isinstance(llm, ChatGroq):
        system_message = SystemMessage(content="Tu es un assistant académique qui aide les étudiants avec des informations sur les formations universitaires.")
        human_message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([system_message, human_message])
        answer = response.content
    else:
        # Fallback pour d'autres types de LLM
        answer = llm(formatted_prompt)
    
    # Ajouter les sources
    sources = [doc.metadata.get("source", "Source inconnue") for doc in docs]
    sources_str = "Sources: " + ", ".join(sources)
    
    # Retourner la réponse complète
    return answer + "\n\n" + sources_str

# Interface utilisateur simple
def chatbot():
    print("Bienvenue dans le chatbot RAG académique avec Groq! Posez vos questions sur les formations universitaires.")
    print("(Tapez 'exit' pour quitter)")
    
    # Afficher le modèle utilisé
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