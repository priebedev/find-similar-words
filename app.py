# --- Imports ---
import streamlit as st
import numpy as np
import requests
from pathlib import Path
from numpy.linalg import norm
import random
import nltk
from nltk.corpus import stopwords
import os

# Tell NLTK to use the local data folder
nltk.data.path.append("nltk_data")

# Set NLTK data path to a temp folder that Streamlit Cloud can write to
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK models to that folder
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

# --- Streamlit config ---
st.set_page_config(page_title="Find Similar Words!!!", layout="centered")

# --- First-load spinner rerun ---
if "first_load" not in st.session_state:
    st.session_state.first_load = True
    st.rerun()

# --- NLTK setup ---
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

# --- Load common words ---
@st.cache_data
def load_common_words():
    with open("common_words.txt", "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

common_words = load_common_words()

# --- Load GloVe embeddings ---
@st.cache_resource
def load_glove_embeddings(url):
    embeddings = {}
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 51:  # 1 word + 50 dimensions
            continue  # Skip lines that aren't valid embeddings
        word = parts[0]
        try:
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
        except ValueError:
            continue  # Skip any line where conversion to float fails
    return embeddings

glove_url = st.secrets["glove_url"]

with st.spinner("🔄 Loading word embeddings..."):
    glove = load_glove_embeddings(glove_url)

# --- Clear first load flag after load ---
if st.session_state.get("first_load"):
    st.session_state.first_load = False
    st.rerun()

# --- Helper functions ---
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def is_valid_word(word):
    return (
        len(word) > 2 and
        word.isalpha() and
        word.lower() not in stop_words
    )

def get_pos_tag(word):
    tag = nltk.pos_tag([word])[0][1]
    return {
        'NN': 'noun',
        'NNS': 'plural noun',
        'VB': 'verb',
        'VBD': 'past tense verb',
        'VBG': 'gerund verb',
        'VBN': 'past participle',
        'VBP': 'verb (non-3rd person)',
        'VBZ': 'verb (3rd person)',
        'JJ': 'adjective',
        'JJR': 'comparative adj',
        'JJS': 'superlative adj',
        'RB': 'adverb',
        'RBR': 'comparative adv',
        'RBS': 'superlative adv'
    }.get(tag, tag)

def get_similar_words(word, embeddings, top_n=5):
    if word not in embeddings:
        return []
    vec = embeddings[word]
    similarities = []
    for other_word, other_vec in embeddings.items():
        if other_word == word or not is_valid_word(other_word):
            continue
        sim = cosine_similarity(vec, other_vec)
        similarities.append((other_word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# --- UI ---
st.title("🔍 Find Similar Words")
st.markdown("Enter a word to find similar words!")

top_n = st.slider("How many similar words would you like to see?", 3, 50, 5)

# --- Word input ---
query = st.text_input("Word", value=st.session_state.get("query", ""), placeholder="Type a word...", label_visibility="visible")
st.components.v1.html("""
    <script>
        window.onload = function() {
            const el = window.parent.document.querySelector('input[type="text"]');
            if (el) el.focus();
        }
    </script>
""", height=0)

# --- Random Word Button ---
if st.button("🎲 Random Word"):
    with st.spinner("Picking a random word..."):
        candidate_words = [
            w for w in glove
            if is_valid_word(w) and w in common_words
        ]
        st.session_state.query = random.choice(candidate_words)
        st.rerun()

# --- Search and Display ---
if query:
    query = query.lower()
    st.session_state.query = query

    if "history" not in st.session_state:
        st.session_state.history = []
    if query not in st.session_state.history and query in glove:
        st.session_state.history.insert(0, query)
        st.session_state.history = st.session_state.history[:10]

    if query not in glove:
        st.error(f"Sorry, the word '{query}' is not in the vocabulary.")
    else:
        with st.spinner("Searching for similar words..."):
            results = get_similar_words(query, glove, top_n=top_n)
        st.subheader(f"Words similar to '{query}':")
        for word, score in results:
            pos = get_pos_tag(word)
            st.write(f"• **{word}** *(POS: {pos})* — similarity score: {score:.4f}")

# --- Sidebar: history ---
with st.sidebar:
    st.markdown("### 🔁 Search History")
    for word in st.session_state.get("history", []):
        st.write(f"- {word}")


