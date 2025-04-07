<h1 align="center">🔍 Pic2Vec 🔍</h1>
<h2 align="center">An Unsupervised Image Search Engine using Deep Visual Embeddings</h2>

<p align="center">
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-App-red" alt="Streamlit">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-Framework-orange" alt="PyTorch">
  </a>
  <a href="https://github.com/facebookresearch/faiss">
    <img src="https://img.shields.io/badge/FAISS-Search-blue" alt="FAISS">
  </a>
</p>

<div align="center">

Allow users to query using an image and return the **most similar** images.

Demonstrate **unsupervised** image retrieval techniques.

Serve as a foundation for scalable, real-world **image search systems**.

</div>

---

<h1 align="center">How it Works</h1>

### **Dataset**
This system is built using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 

### **Embedding Generation**
- Each image in the CIFAR-10 dataset is passed through a **pre-trained ResNet18 model**.
- **ResNet18** is a **Convolutional Neural Network (CNN)** architecture with 18 layers:
  - [HERE](https://huggingface.co/microsoft/resnet-18) is the model card.
  - Uses **skip connections** to tackle the **vanishing gradient problem**.
- These embeddings capture key visual features like shape, color, and texture.
- We remove the final classification layer (to make this **unnanotated**).
- The embeddings are **L2-normalized** to ensure that similarity comparisons use cosine similarity effectively.
  - L2 normalization scales each embedding vector to have a length (magnitude) of 1.
  - We want capture the "direction" of embeddings, not their length (the direction of the vector contains meaning).
  - If we don't normalize:
    - Distances could be influenced by vector length.
    - Similar images could appear far apart in embedding space due to magnitude differences.

### **FAISS**
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

### **Evaluation**
1. **Precision@k:** Measures how many of the top-k retrieved images belong to the same class as the query image.
    - This the **supervised** evaluation.
    - Formula: `Precision@k = (Number of correctly retrieved images) ÷ k`
      - `k` = number of retrieved images
      - Correctly retrieved images = retrieved images whose label matches the query image’s label
    - Example:
      - If you query with an image of a **cat** and retrieve 5 images with labels: `[cat, cat, dog, cat, airplane]`
      - `Precision@5 = 3 / 5 = 0.6`
2. **k-means clustering with t-SNE**: Group the embeddings into 10 classes using k-means clustering, then use t-SNE to visualize.
    - Use k-means to find `n_clusters=10` groups in the 512-dimensional image embeddings.
    - Use t-SNE to compress 512-dimensional embeddings into 2 dimensions so we can plot them.
3. **Qualitative Evaluation:** The Streamlit app allows you to visually inspect retrieved results to assess visual similarity.

---

<h1 align="center">File Structure</h1>

```
.
├── README.md
├── app
│   └── app.py
├── cifar
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── notebooks
│   └── image_search.ipynb
├── outputs
│   ├── embeddings.npy
│   ├── faiss_index.index
│   └── labels.npy
└── requirements.txt
```
