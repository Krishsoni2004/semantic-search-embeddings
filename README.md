# 🔍 Semantic Search using Hugging Face Embeddings

This project demonstrates a simple **semantic search engine** using Hugging Face sentence embeddings and cosine similarity.

Instead of keyword matching, it finds the **meaning-based most relevant document** for a query.

---

## 🚀 Features

- Convert text into embeddings using HuggingFace models
- Convert query into vector representation
- Compute cosine similarity between query and documents
- Return the most semantically similar result

---

## 🧠 How it works

1. Load pre-trained embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
2. Convert documents into vector embeddings
3. Convert user query into embedding
4. Compare similarity using cosine similarity
5. Return the best matching document

---

## 🛠️ Tech Stack

- Python 🐍
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- NumPy

---

## 📂 Project Structure
