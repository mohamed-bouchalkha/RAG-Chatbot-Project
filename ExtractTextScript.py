import os
import fitz  # PyMuPDF

# Chemin vers ton dossier PDF
pdf_dir = r"C:\Users\hp\Desktop\S2-WISD MASTER\RO\RAG-Chatbot-Project\DataChatbot"
output_dir = os.path.join(pdf_dir, "extracted_texts")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Sauvegarder le texte dans un fichier .txt
        base_name = os.path.splitext(filename)[0]
        with open(os.path.join(output_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[✓] Fichier traité : {filename}")
