from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Setup Flask
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Détection automatique de l'environnement
if os.path.exists("/app"):  # Dans Docker
    embeddings_dir = "/app/embeddings"
else:  # Sur Windows en développement
    embeddings_dir = os.path.join(os.path.dirname(__file__), "embeddings")

print(f"📁 Utilisation du répertoire embeddings: {embeddings_dir}")

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
    embeddings_file = os.path.join(embeddings_dir, "embeddings_data.pkl")
    print(f"🔍 Recherche du fichier: {embeddings_file}")
    
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)
    
    embeddings_list = data["embeddings"]
    texts = data["texts"]
    metadatas = data["metadatas"]
    
    print(f"✅ Embeddings chargés avec succès! ({len(texts)} documents)")
except Exception as e:
    print(f"❌ Erreur lors du chargement des embeddings: {str(e)}")
    print(f"📂 Contenu du dossier {embeddings_dir}:")
    if os.path.exists(embeddings_dir):
        for file in os.listdir(embeddings_dir):
            print(f"  - {file}")
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
    print("✅ Modèle d'embedding HuggingFace chargé!")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement du modèle d'embedding: {str(e)}")
    
    # Charger l'alternative TF-IDF
    try:
        tfidf_file = os.path.join(embeddings_dir, "tfidf_embeddings.pkl")
        with open(tfidf_file, "rb") as f:
            tfidf_data = pickle.load(f)
        
        vectorizer = tfidf_data["vectorizer"]
        embeddings_list = tfidf_data["embeddings"]
        texts = tfidf_data["texts"]
        metadatas = tfidf_data["metadatas"]
        embeddings_list = np.array(embeddings_list)
        embeddings_list /= np.linalg.norm(embeddings_list, axis=1, keepdims=True)   

        class TfidfEmbedder:
            def __init__(self, vectorizer):
                self.vectorizer = vectorizer
            
            def embed_query(self, query):
                """Encoder une requête avec TF-IDF"""
                query_vec = self.vectorizer.transform([query]).toarray()[0]
                return query_vec
        
        embedding_model = TfidfEmbedder(vectorizer)
        print("✅ Modèle TF-IDF de secours chargé!")
    except Exception as tfidf_error:
        print(f"❌ Impossible de charger l'embedder TF-IDF: {tfidf_error}")
        exit(1)

# Créer le retriever personnalisé
retriever = CustomRetriever(embeddings_list, texts, metadatas, embedding_model)

# Template pour le prompt
template = """
Tu es un assistant intelligent pour les étudiants universitaires Et Visiteur de La Ville De Fes.

Tu as deux rôles principaux :
1. Tu es un assistant académique spécialisé dans les programmes de formation, brochures, règlements et autres documents universitaires.
2. Tu es aussi un guide local de confiance pour la ville de Fès : tu peux fournir des informations touristiques, pratiques, culturelles et historiques sur la ville, ses quartiers, ses monuments, ses moyens de transport, etc.

Exemple de question :
C'est qui Jamal RIFFI ?
response :
Coordinateur de master MLAIM.
Donne moi le num De tele pour contactes Pr. Jamal RIFF?
response : Vous pouvez le contacter par email à mastermlaim@gmail.com ou par téléphone au 05 35 73 33 49 ou 05 35 73 30 59.
=== QUESTION DE L'ÉTUDIANT ===
{question}
 
=== DOCUMENTS FOURNIS (CONTEXTE) ===
{context}

RÈGLES IMPORTANTES :
- Donne des réponses COURTES et PRÉCISES (4 phrases maximum)
- Ne réponds **que** à la question posée
- Ne donne **aucune information supplémentaire**
- Ne donne pas d'informations supplémentaires non demandées
- Utilise des phrases simples et directes
- Si la question est simple, réponds simplement
- Évite les longues explications
Réponse :
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

# Configuration de Groq
try:
    # Récupérer la clé API depuis les variables d'environnement ou utiliser la valeur par défaut
    groq_api_key = os.getenv("GROQ_API_KEY", "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")
    
    # Initialiser le modèle Groq
    llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        max_tokens=1000,
        api_key=groq_api_key
    )
    
    print(f"✅ Modèle Groq initialisé avec succès!")
    
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation de Groq: {str(e)}")
    print("🔄 Utilisation d'une approche simplifiée...")
    
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
    
    return answer

# Endpoint de base pour vérifier que l'API fonctionne
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "RAG Chatbot API is running!"})

# Endpoint for answering questions
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("question", "")
        if not query:
            return jsonify({"error": "No question provided."}), 400
        
        print(f"📝 Question reçue: {query}")
        answer = process_query(query)
        print(f"✅ Réponse générée: {answer[:100]}...")
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Démarrage de l'API RAG Chatbot...")
    app.run(debug=True, host="0.0.0.0", port=5000)