import os
import re
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chemin vers le dossier contenant les fichiers texte extraits
input_dir = r"C:\Users\hp\Desktop\RAG-Chatbot-Project\DataChatbot\extracted_texts"
output_dir = r"C:\Users\hp\Desktop\RAG-Chatbot-Project\DataChatbot\embeddings"
os.makedirs(output_dir, exist_ok=True)

# Fonction pour nettoyer le texte
def clean_text(text):
    # Supprimer les caractères de contrôle et les caractères non imprimables
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les lignes vides
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Supprimer les en-têtes et pieds de page communs
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Supprimer les numéros de page isolés
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    # Supprimer les URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()

# Liste pour stocker tous les documents traités
all_documents = []

# Traiter chaque fichier texte
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        print(f"Traitement du fichier : {filename}")
        
        try:
            # Charger le contenu du fichier
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Nettoyer le texte
            cleaned_content = clean_text(content)
            
            # Sauvegarder le texte nettoyé (optionnel)
            cleaned_file_path = os.path.join(output_dir, f"cleaned_{filename}")
            with open(cleaned_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            
            # Ajouter à notre liste de documents avec métadonnées
            all_documents.append({
                "content": cleaned_content,
                "metadata": {"source": filename}
            })
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {str(e)}")

# Diviser les documents en chunks pour un meilleur traitement
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Transformer nos documents en format compatible avec LangChain
docs = []
for doc in all_documents:
    chunks = text_splitter.create_documents(
        texts=[doc["content"]],
        metadatas=[doc["metadata"]]
    )
    docs.extend(chunks)

print(f"Nombre total de chunks créés : {len(docs)}")

# Extraction des textes des chunks pour créer les embeddings
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

try:
    # Solution 1: Essayer d'utiliser HuggingFace BGE Embeddings directement sans FAISS
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Créer et sauvegarder les embeddings manuellement
    print("Création des embeddings...")
    embeddings = embedding_model.embed_documents(texts)
    
    # Sauvegarde manuelle sans FAISS
    embeddings_data = {
        "embeddings": embeddings,
        "texts": texts,
        "metadatas": metadatas
    }
    
    with open(os.path.join(output_dir, "embeddings_data.pkl"), "wb") as f:
        pickle.dump(embeddings_data, f)
    
    print("Embeddings sauvegardés avec succès dans embeddings_data.pkl")
    
    # Essayer d'installer FAISS automatiquement (optionnel)
    try:
        import subprocess
        print("Tentative d'installation de FAISS...")
        subprocess.check_call(["pip", "install", "faiss-cpu"])
        print("FAISS installé avec succès, vous pouvez relancer le script pour l'utiliser")
    except:
        print("Impossible d'installer FAISS automatiquement. Pour l'installer manuellement:")
        print("pip install faiss-cpu")
    
except Exception as e:
    print(f"Erreur avec HuggingFaceBgeEmbeddings: {str(e)}")
    
    try:
        # Solution 2: Utiliser scikit-learn pour les embeddings en cas d'échec
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Utilisation de TF-IDF comme solution de secours...")
        vectorizer = TfidfVectorizer(max_features=1024)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Convertir sparse matrix en array dense pour faciliter la sauvegarde
        embeddings = tfidf_matrix.toarray()
        
        # Sauvegarde des embeddings TF-IDF
        embeddings_data = {
            "embeddings": embeddings,
            "texts": texts,
            "metadatas": metadatas,
            "vectorizer": vectorizer  # Sauvegarde du vectorizer pour transformations futures
        }
        
        with open(os.path.join(output_dir, "tfidf_embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings_data, f)
        
        print("Embeddings TF-IDF sauvegardés avec succès dans tfidf_embeddings.pkl")
        
    except Exception as e:
        print(f"Erreur avec la solution de secours TF-IDF: {str(e)}")
        print("Pour résoudre les problèmes d'installation, essayez:")
        print("1. pip install scikit-learn")
        print("2. pip install faiss-cpu")
        print("3. pip install langchain-huggingface")