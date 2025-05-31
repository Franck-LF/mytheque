
from os import listdir
from os.path import isdir, isfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pymupdf

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

import faiss

import streamlit as st



path = "BOOKS/"
document_paths = []
titles = []
documents = []
counter = 0

# Extract raw text fro mpdf
for item in listdir(path):
    counter +=1
    print(item)
    txt = pymupdf.get_text(path + item)
    if txt:
        if counter == 1:
           print(txt)
        assert len(txt) == 1
        titles.append(item)
        documents.append(txt[0])
    else:
        print(f"ERROR on file: {item}")

df = pd.DataFrame({"title" : titles, "document" : documents})
print(df.shape[0])

# Load model
model = SentenceTransformer(r"C:\Users\Utilisateur\Documents\mytheque\Models\sentence-camembert-large", local_files_only=True)

# Text embedding
def embedding(txt):
    temp = model.encode(txt)
    # print(type(temp))
    # print(type(temp.tolist()))
    # print(type(temp.item(0)))
    return temp

df['embedding'] = df['document'].apply(embedding)
df['embedding'] = df['embedding'].apply(lambda emb : emb.tolist())
# Careful, it displays th edataframe in streamlit inteface
#df['embedding']

# Format embeddings to avoid np.array(np.array(), ... np.array())
embeddings = []
for item in df['embedding']:
    embeddings.append(item)
embeddings = np.array(embeddings)
print(embeddings.shape)

dim = embeddings.shape[1]
assert dim == 1024
print("dim", dim)
index = faiss.IndexFlatIP(dim)
print(type(index))
index.add(embeddings)


if __name__ == "__main__":
  
  st.title("Moteur de Recherche")
  # Text Input
  search_text = st.text_area("Enter Text:", "", key="text1")
  st.radio("", ["Mots exacts", "Recherche sémantique"], index = 1, horizontal = False)
  test = st.text_input("Test")

  # with st.sidebar:
  #   display_req_url = st.checkbox("Display Request URL") # , value = True)
  #   display_geoloc = st.checkbox("Display Geolocation")
  #   display_stMap = st.checkbox("Display st.map()")
  #   st.write("API")

  if st.button('Search', type="primary", icon="🔍"): 
    st.write(search_text)
    # search_vector = model.encode([search_text])
    # distances, indices = index.search(search_vector, k=10)
    # st.subheader("Top résultats :")
    # st.markdown(f"Result {indices}")
    # st.divider()
    
  else:
     st.write("Nothing")




