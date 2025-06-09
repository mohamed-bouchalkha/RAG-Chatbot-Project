# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Exposer le port 5000
EXPOSE 5000

# Définir les variables d'environnement
ENV FLASK_APP=chatbot_api.py
ENV FLASK_ENV=production

# Commande pour démarrer l'application
CMD ["python", "chatbot_api.py"]