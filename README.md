# Image Detection using CNN–ViT–GNN Hybrid Model

## 📌 Overview
This project implements an **image detection/classification framework** using a hybrid deep learning architecture that combines:

- **Convolutional Neural Networks (CNNs)** for local feature extraction  
- **Vision Transformers (ViTs)** for global contextual understanding  
- **Graph Neural Networks (GNNs)** for explicit relational reasoning between image regions  

The motivation is to go beyond traditional CNN-based pipelines by modeling **relationships between image patches as a graph**, which CNNs and ViTs alone do not explicitly capture.

---

## 🧠 Architecture Motivation

| Component | Role |
|---------|------|
| CNN | Captures low-level spatial features (edges, textures) |
| ViT | Models long-range/global dependencies |
| GNN | Learns relationships between image regions (graph nodes) |

### Overall Pipeline
Input Image
↓
CNN Backbone
↓
Patch / Token Embeddings
↓
Vision Transformer
↓
Graph Construction (patches as nodes)
↓
Graph Neural Network
↓
Prediction