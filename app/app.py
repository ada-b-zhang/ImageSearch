import streamlit as st
import numpy as np
import faiss
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------------------------

DATA_DIR = "../cifar"
OUTPUT_DIR = "../outputs"
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ----------------------------------------------------------------------------------------------------
# LOAD DATA & INDEX
# ----------------------------------------------------------------------------------------------------

@st.cache_data
def load_data():
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
    embeddings = np.load(f"{OUTPUT_DIR}/embeddings.npy")
    labels = np.load(f"{OUTPUT_DIR}/labels.npy")
    index = faiss.read_index(f"{OUTPUT_DIR}/faiss_index.index")
    return data, embeddings, labels, index

data, embeddings, labels, index = load_data()

# ----------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------------------------

def unnormalize(img):
    return img.permute(1, 2, 0) * 0.5 + 0.5

def search(query_idx, k=5):
    query_embedding = embeddings[query_idx].reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

# ----------------------------------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------------------------------

st.title("🔍 Pic2Vec")
st.write("An Unsupervised Image Search Engine using Deep Visual Embeddings.")

query_option = st.radio("Select query image:", ["Random", "By Index"])
k = st.slider("Number of similar images to retrieve (k):", min_value=1, max_value=10, value=5)

if query_option == "Random":
    query_idx = np.random.randint(0, len(data))
else:
    query_idx = st.number_input("Enter query image index:", min_value=0, max_value=len(data)-1, value=0)

if st.button("🔍 Search"):
    st.subheader(f"Query Image (Index: {query_idx})")
    query_img, query_label = data[query_idx]
    st.image(unnormalize(query_img).numpy(), caption=f"Label: {query_label} ({class_names[query_label]})", width=100)

    distances, indices = search(query_idx, k)
    
    st.subheader(f"Top {k} Similar Images")
    cols = st.columns(k)
    for i in range(k):
        img_idx = indices[i]
        img, label = data[img_idx]
        with cols[i]:
            st.image(unnormalize(img).numpy(), caption=f"Label: {label} ({class_names[label]})\nDistance: {distances[i]:.2f}", width=100)
