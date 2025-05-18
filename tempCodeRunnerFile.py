    if not groq_api_key:
        # Si la clé n'est pas dans les variables d'environnement, demander à l'utilisateur
        groq_api_key = input("Veuillez entrer votre clé API Groq: ")
        os.environ["GROQ_API_KEY"] = groq_api_key