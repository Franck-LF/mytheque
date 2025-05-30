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

