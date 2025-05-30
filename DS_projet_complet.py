import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
PDF_DIR = "MYTHS/"  # Remplacez par votre dossier de PDF
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 512  # Taille des fragments de texte

# 1. Extraction des PDF
def extract_pdf_text(pdf_path):
    """Extrait le texte d'un PDF avec découpage en pages"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""  # Gère les pages vides
            text += page_text + "\n\n"
    return text.strip()

# 2. Découpage en chunks
def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Découpe le texte en fragments sémantiques"""
    words = text.split()
    chunks = []
    current_chunk = []
    char_count = 0
    
    for word in words:
        if char_count + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            char_count += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            char_count = len(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# 3. Chargement des documents
@st.cache_resource
def load_documents():
    """Charge et préprocesse tous les PDFs"""
    documents = []
    metadata = []
    
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            text = extract_pdf_text(path)
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "filename": filename,
                    "page": i + 1,
                    "chunk": chunk[:100] + "..."  # Extrait pour affichage
                })
    
    return documents, metadata

# 4. Modèle d'embedding
@st.cache_resource
def load_model():
    """Charge le modèle Sentence-BERT"""
    return SentenceTransformer(MODEL_NAME)

# 5. Indexation FAISS
@st.cache_resource
def create_faiss_index(_model, documents):
    """Crée l'index de recherche vectorielle"""
    embeddings = _model.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)  # Normalisation pour cosine similarity
    index.add(embeddings)
    return index, embeddings

# Interface Streamlit
st.title("🔍 Moteur de Recherche Sémantique de PDF")
st.write(f"Base documentaire: {len(os.listdir(PDF_DIR))} PDFs")

# Chargement des données
documents, metadata = load_documents()
model = load_model()
faiss_index, embeddings = create_faiss_index(model, documents)

# Recherche
query = st.text_input("Entrez votre recherche:", 
                      "cinéma abrutissant clichés castagnettes")

if st.button("Rechercher") or query:
    # Embedding de la requête
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Recherche FAISS
    D, I = faiss_index.search(query_embedding, k=5)
    
    # Affichage des résultats
    st.subheader("Top 5 résultats:")
    for i, (score, idx) in enumerate(zip(D[0], I[0])):
        doc_meta = metadata[idx]
        st.markdown(f"""
        ### 📄 Résultat #{i+1} (Score: {score:.3f})
        **Fichier:** {doc_meta['filename']}  
        **Extrait:**  
        > {doc_meta['chunk']}
        """)
        st.divider()

# Fonction bonus: comparaison TF-IDF vs Embeddings
if st.checkbox("Comparer avec TF-IDF"):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_tfidf = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_tfidf_idx = np.argsort(tfidf_scores)[-5:][::-1]
    
    st.subheader("Résultats TF-IDF:")
    for i, idx in enumerate(top_tfidf_idx):
        doc_meta = metadata[idx]
        st.write(f"#{i+1} [{tfidf_scores[idx]:.3f}] {doc_meta['filename']}")

# Exécuter avec: streamlit run nom_du_script.py




# Améliorations possibles
# 
# découpage avec Langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_text(text)

# Cache des embeddings

# Sauvegarder les embeddings
np.save("doc_embeddings.npy", embeddings)

# Charger ultérieurement
embeddings = np.load("doc_embeddings.npy")



# Pour comparer TF-IDF et embedding

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau

def compare_similarities(query, top_k=10):
    """Compare les résultats TF-IDF vs Embeddings pour une requête"""
    # 1. Calcul TF-IDF
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_tfidf_idx = np.argsort(tfidf_scores)[-top_k:][::-1]
    
    # 2. Calcul Embeddings
    query_embed = model.encode([query])
    faiss.normalize_L2(query_embed)
    D, I = faiss_index.search(query_embed, top_k)
    embedding_scores = D[0]
    embedding_idx = I[0]
    
    # 3. Création du DataFrame comparatif
    results = []
    for i, (idx_tf, score_tf) in enumerate(zip(top_tfidf_idx, tfidf_scores[top_tfidf_idx])):
        # Recherche position dans les résultats embeddings
        try:
            pos_emb = np.where(embedding_idx == idx_tf)[0][0]
            score_emb = embedding_scores[pos_emb]
        except:
            score_emb = 0.0
            pos_emb = top_k + 1
            
        results.append({
            "Document": metadata[idx_tf]['filename'],
            "TF-IDF Rank": i + 1,
            "TF-IDF Score": score_tf,
            "Embedding Rank": pos_emb + 1,
            "Embedding Score": score_emb,
            "In Both Top": int(pos_emb < top_k)
        })
    
    df = pd.DataFrame(results)
    
    # 4. Métriques statistiques
    corr_pearson, _ = pearsonr(df["TF-IDF Score"], df["Embedding Score"])
    corr_kendall, _ = kendalltau(df["TF-IDF Rank"], df["Embedding Rank"])
    overlap = df["In Both Top"].mean() * 100
    
    # 5. Visualisations
    plt.figure(figsize=(15, 6))
    
    # Scatterplot comparatif
    plt.subplot(121)
    sns.scatterplot(data=df, x="TF-IDF Score", y="Embedding Score", hue="In Both Top", s=100)
    plt.title(f"Corrélation des scores (Pearson: {corr_pearson:.2f})")
    plt.plot([0,1], [0,1], 'r--', alpha=0.3)
    
    # Diagramme de classement
    plt.subplot(122)
    df_melt = df.melt(id_vars="Document", value_vars=["TF-IDF Rank", "Embedding Rank"], 
                     var_name="Méthode", value_name="Rank")
    sns.lineplot(data=df_melt, x="Document", y="Rank", hue="Méthode", marker="o")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Overlap top-{top_k}: {overlap:.1f}% | Kendall Tau: {corr_kendall:.2f}")
    plt.tight_layout()
    
    return df, plt

# Dans l'interface Streamlit (ajouter après la recherche principale)
if st.button("Comparer TF-IDF vs Embeddings"):
    st.subheader("Analyse comparative des méthodes")
    df, plot = compare_similarities(query, top_k=10)
    st.pyplot(plot)
    st.dataframe(df.sort_values("TF-IDF Rank"))