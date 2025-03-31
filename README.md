# ImageSearch: Image Retrieval with CIFAR-10

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Search-blue)](https://github.com/facebookresearch/faiss)


---
This project implements an **image similarity search engine** using the **CIFAR-10 dataset**. Given a query image, the system retrieves the top-k most visually similar images from a large, **unannotated** database.

## Project Overview

The goal of this project is to build a system that can:
- **Index and search images based on visual similarity** without relying on labels.
- Allow users to query using an image and return the most similar images.
- Demonstrate **unsupervised image retrieval** techniques.
- Serve as a foundation for scalable, real-world image search systems.

---

## How it Works

### 1. **Embedding Generation**
- Each image in the CIFAR-10 dataset is passed through a **pre-trained ResNet18 model**.
- **ResNet18** is a **Convolutional Neural Network (CNN)** architecture with 18 layers: [HERE](https://huggingface.co/microsoft/resnet-18) is the model card
- These embeddings capture key visual features like shape, color, and texture.
- We remove the final classification layer (to make this **unnanotated**).
- The embeddings are **L2-normalized** to ensure that similarity comparisons use cosine similarity effectively.
  - L2 normalization scales each embedding vector to have a length (magnitude) of 1.
  - We want capture the "direction" of embeddings, not their length (the direction of the vector contains meaning).
  - If we don't normalize:
    - Distances could be influenced by vector length.
    - Similar images could appear far apart in embedding space due to magnitude differences.

### 2. **FAISS**
- FAISS is "a library for efficient similarity search and clustering of dense vectors." [Source](https://github.com/facebookresearch/faiss)
- All embeddings are stored in a **FAISS index** (`IndexFlatL2`):
  1. It calculates the L2 distance between your query embedding and every stored embedding.
  2. Sorts them.
  3. Returns:
     - `index`: The indices of the top-k closest vectors (these correspond to image indices in your dataset).
     - `distance`: The L2 distance values.
- This is a nearest-neighbor-like search.
- When a user provides a query image, the app retrieves the **top-k most similar images** based on embedding distance.
- TLDR:
  - It computes distances between the query embedding and all stored embeddings.
  - It returns the indices of the k most similar embeddings.

### 3. **Evaluation**
Although the system is **unsupervised**, evaluation is done using CIFAR-10 class labels:
- **Precision@k:** Measures how many of the top-k retrieved images belong to the same class as the query image.
  - **Formula:** Precision@k = (Number of correctly retrieved images) ÷ k
    - k = number of retrieved images
    - Correctly retrieved images = retrieved images whose label matches the query image’s label
  - Example:
    - If you query with an image of a **cat** and retrieve 5 images with labels: `[cat, cat, dog, cat, airplane]`
    - Precision@5 = 3 / 5 = 0.6
- **Qualitative Evaluation:** The Streamlit app allows you to visually inspect retrieved results to assess visual similarity.

---
