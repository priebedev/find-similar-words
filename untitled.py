import streamlit as st
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import string
from nltk.corpus import stopwords
import nltk

# Load stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def is_valid_word(word):
    return (
        len(word) > 2 and
        word.isalpha() and
        word.lower() not in stop_words
    )

def get_similar_words(word, embeddings, top_n=5):
    if word not in embeddings:
        return []
    vec = embeddings[word]
    similarities = []
    for other_word, other_vec in embeddings.items():
        if other_word == word:
            continue
        if not is_valid_word(other_word):
            continue
        sim = cosine_similarity(vec, other_vec)
        similarities.append((other_word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- UI ---
st.set_page_config(page_title="GloVe Thesaurus", layout="centered")
st.title("🔍 GloVe-Powered Thesaurus")

# Load the model
with st.spinner("Loading word embeddings..."):
    base_path = Path("C:/Users/drumw/Dev Projects/thesaurus/glove.6B")
    file_path = base_path / "glove.6B.100d.txt"
    glove = load_glove_embeddings(file_path)

# Input
query = st.text_input("Enter a word:", "")

# Show results
if query:
    if query not in glove:
        st.error(f"Sorry, the word '{query}' is not in the vocabulary.")
    else:
        results = get_similar_words(query.lower(), glove)
        st.subheader(f"Words similar to '{query}':")
        for word, score in results:
            st.write(f"• **{word}** — similarity score: {score:.4f}")
