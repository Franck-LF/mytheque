import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import spacy
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

import faiss

import streamlit as st


if __name__ == "__main__":
  
  st.write('First texte')
  st.title("🔍 Moteur de Recherche Sémantique de PDF")
  st.write('Base documentaire')

  with st.sidebar:
    display_req_url = st.checkbox("Display Request URL") # , value = True)
    display_geoloc = st.checkbox("Display Geolocation")
    display_stMap = st.checkbox("Display st.map()")
    st.write("API")

  destination = st.text_input("Destination", "Saisir votre recherche")
  st.radio("", ["Mots-clés", "Recherche sémantique"], index = 1, horizontal = False)