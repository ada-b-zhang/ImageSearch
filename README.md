# ImageSearch: Image Retrieval with CIFAR-10

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Search-blue)](https://github.com/facebookresearch/faiss)

This project implements an **image similarity search engine** using the **CIFAR-10 dataset**. Given a query image, the system retrieves the top-k most visually similar images from a large, **unannotated** database.

> **Technologies used:** PyTorch, FAISS, Streamlit

---

## Project Overview

The goal of this project is to build a system that can:
- **Index and search images based on visual similarity** without relying on labels.
- Allow users to query using an image and return the most similar images.
- Demonstrate **unsupervised image retrieval** techniques.
- Serve as a foundation for scalable, real-world image search systems.

---

## How it works

### 1. **Embedding Generation**
- Each image in the CIFAR-10 dataset is passed through a **pre-trained ResNet18 model**.
- The model’s classification head is removed, and the output of the last convolutional layer is used as a **512-dimensional image embedding**.
- These embeddings capture key visual features like shape, color, and texture.
- The embeddings are **L2-normalized** to ensure that similarity comparisons use cosine similarity effectively.

### 2. **FAISS Index (Similarity Search Engine)**
- All embeddings are stored in a **FAISS index** (`IndexFlatL2`).
- This index enables fast and efficient **nearest neighbor search**.
- When a user provides a query image, the app retrieves the **top-k most similar images** based on embedding distance.

### 3. **Evaluation**
Although the system is **unsupervised**, evaluation is done using CIFAR-10 class labels:
- **Precision@k:** Measures how many of the top-k retrieved images belong to the same class as the query image.
- **Qualitative Evaluation:** The Streamlit app allows you to visually inspect retrieved results to assess visual similarity.

---
