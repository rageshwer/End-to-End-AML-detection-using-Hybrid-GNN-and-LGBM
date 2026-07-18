# 🛡️ AML Graph Intelligence Platform

### End-to-End Anti-Money Laundering Detection using Hybrid Graph Neural Networks, Explainable AI & MLOps

An enterprise-grade **Anti-Money Laundering (AML)** platform that detects suspicious cryptocurrency transactions by combining **Graph Neural Networks**, **Gradient Boosting**, and **Explainable AI**.

Rather than evaluating each transaction independently, the platform learns from the **structure of the Bitcoin transaction network**, enabling it to identify laundering patterns hidden across multiple connected transactions.

The project demonstrates the complete lifecycle of a production-ready Graph Machine Learning system—from temporal graph construction and hybrid GNN modeling to explainable AI, cloud deployment, performance monitoring, and MLOps.

---

#  Live Demo

**Interactive Demo**

http://13.232.88.197:8501/

> **Note**
>
> The application evaluates transactions from the unseen test period (**Timesteps 35–49**) using pre-computed graph embeddings generated during the leak-proof training pipeline.
>
> Simply enter a transaction ID (0–16669) to obtain:
>
> - AML Risk Probability
> - SHAP Explainability
> - Graph Contribution Analysis
> - Performance Monitoring Dashboard

---

#  Project Highlights

| Category | Details |
|-----------|---------|
| **Model** | Hybrid GraphSAGE + GAT + LightGBM |
| **Evaluation** | Strict Chronological (Leak-Proof) |
| **PR-AUC** | **0.7158** |
| **Illicit F1** | **0.6932** |
| **Precision (Illicit)** | **0.87** |
| **Recall (Illicit)** | **0.58** |
| **Explainability** | SHAP Waterfall |
| **Monitoring** | Evidently AI |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Deployment** | Docker + AWS EC2 + Amazon ECR |
| **Experiment Type** | Temporal Graph Machine Learning |

---

#  Application Preview

## Landing Page

The application introduces analysts to the complete AML workflow—from graph construction through explainable inference and performance monitoring.

Features include:

- Graph Intelligence Overview
- Chronological Training Pipeline
- Hybrid Graph Architecture
- Real-Time Transaction Analysis

<img width="1876" height="893" alt="Screenshot 2026-07-10 at 6 13 56 PM" src="https://github.com/user-attachments/assets/47e6fab3-52b6-4e33-a669-135d1d843c0e" />
<img width="1852" height="853" alt="Screenshot 2026-07-10 at 6 14 08 PM" src="https://github.com/user-attachments/assets/10fe502d-12bb-4b54-a46c-4834e949df83" />


---

## Real-Time AML Risk Assessment

Each transaction receives:

- AML Risk Probability
- Decision Recommendation
- Latency Metrics
- Feature-Level Explainability

The SHAP waterfall explains how graph embeddings, local transaction features, and neighborhood aggregation influenced the final prediction.
<img width="1823" height="908" alt="Screenshot 2026-07-10 at 6 28 48 PM" src="https://github.com/user-attachments/assets/43c14601-1226-4fb4-86f9-cce7e8b115bb" />



---

## Model Performance Monitoring

An integrated monitoring dashboard performs an **on-demand Evidently AI audit** using **5,000 randomly sampled transactions** from the unseen test period.

The dashboard reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss
- Confusion Matrix

allowing production performance to be validated against historical benchmarks.

<img width="1874" height="859" alt="Screenshot 2026-07-10 at 6 15 02 PM" src="https://github.com/user-attachments/assets/e8bbe409-ea03-4cfe-9201-3eea2b776353" />


---

#  Business Problem

Financial institutions process millions of transactions every day.

Traditional AML systems typically evaluate each transaction independently using handcrafted rules or tabular machine learning models.

However, money laundering rarely appears within a single transaction.

Instead, suspicious behaviour emerges through:

- Transaction chains
- Multi-hop fund transfers
- Wallet relationships
- Network topology
- Temporal transaction behaviour

These structural patterns cannot be captured using conventional tabular models alone.

This project addresses that challenge by combining **Graph Neural Networks** with **Gradient Boosting** to learn both **network structure** and **transaction-level behaviour**, enabling more effective detection of illicit financial activity.

---

#  Why Graph Neural Networks?

Money laundering is fundamentally a **graph problem**.

Every Bitcoin transaction forms part of a larger financial network where:

- Transactions become graph nodes.
- Bitcoin flows create graph edges.
- Neighboring transactions influence one another.
- Multi-hop relationships reveal hidden laundering patterns.

Unlike traditional machine learning models that evaluate transactions independently, Graph Neural Networks propagate information across connected neighborhoods, allowing each transaction to be evaluated within the context of the surrounding network.

This enables the model to identify structural behaviour that would otherwise remain invisible.

---

#  System Architecture

```text
                    Bitcoin Transaction Graph
                               │
                               ▼

                 Hybrid Graph Neural Network
                (GraphSAGE + Graph Attention)

                               │
                               ▼

                  256-D Graph Embeddings
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼

     165 Engineered Features          Graph Embeddings

               └───────────────┬───────────────┘
                               ▼

                    LightGBM Classifier

                               │
               ┌───────────────┼────────────────┐
               ▼               ▼                ▼

      AML Probability     SHAP Engine     Monitoring

                               │
                               ▼

                      FastAPI Prediction API

                               │
                               ▼

                    Streamlit Analyst Portal

                               │
                               ▼

                Dockerized AWS EC2 Deployment
```

---

# End-to-End Pipeline

```text
Raw Bitcoin Transactions
            │
            ▼

Graph Construction
            │
            ▼

Temporal Graph Split
            │
            ▼

Hybrid GraphSAGE + GAT
            │
            ▼

256-D Graph Embeddings
            │
            ▼

Merge with Local Features
            │
            ▼

LightGBM Classification
            │
            ▼

SHAP Explainability
            │
            ▼

FastAPI
            │
            ▼

Streamlit Dashboard
            │
            ▼

Evidently AI Monitoring
```

---

# Core Capabilities

✅ Hybrid GraphSAGE + GAT Architecture

✅ Leak-Proof Chronological Evaluation

✅ Graph + Tabular Hybrid Learning

✅ Real-Time Transaction Scoring

✅ SHAP Explainability

✅ Evidently AI Performance Monitoring

✅ FastAPI Backend

✅ Streamlit Dashboard

✅ Dockerized Deployment

✅ AWS Cloud Infrastructure

# Dataset Overview

This project is built on the **Elliptic Bitcoin Transaction Dataset**, one of the most widely used benchmark datasets for Anti-Money Laundering (AML) research using Graph Neural Networks.

The dataset represents Bitcoin transactions as a directed graph, where each transaction is connected through the flow of cryptocurrency.

---

## Dataset Characteristics

| Property | Value |
|----------|------:|
| Dataset | Elliptic Bitcoin Transaction Dataset |
| Domain | Cryptocurrency Anti-Money Laundering |
| Graph Type | Directed Transaction Graph |
| Temporal Windows | **49 Timesteps** |
| Node Features | **165 Transaction Features** |
| Classes | Licit / Illicit |
| Learning Task | Binary Node Classification |

Each node represents a **Bitcoin transaction**, while directed edges capture the movement of funds between transactions.

Unlike conventional tabular datasets, every prediction depends not only on a transaction's own attributes but also on the surrounding transaction network.

---

#  Leak-Proof Temporal Evaluation

One of the primary challenges in financial crime detection is **information leakage**.

Random train/test splits allow future transactions to influence past predictions, producing overly optimistic performance estimates that are unrealistic in production systems.

To simulate real-world deployment, this project follows a **strict chronological evaluation strategy**.

| Dataset Split | Timesteps |
|--------------|-----------|
| **Training** | **1 – 28** |
| **Validation** | **29 – 34** |
| **Testing** | **35 – 49 (Unseen)** |

The Graph Neural Network was trained only on historical transactions before generating embeddings for future periods.

This ensures every reported metric reflects performance on transactions that were never observed during training.

---

#  Graph Construction

Each transaction is represented as a node within the Bitcoin transaction network.

```text
Transaction
      │
      ▼

Node Features (165)
      │
      ▼

Directed Bitcoin Flow
      │
      ▼

Transaction Graph
```

The graph captures structural relationships that traditional tabular models cannot represent, allowing information to propagate through neighboring transactions.

---

#  Hybrid Graph Learning Pipeline

Rather than relying solely on Graph Neural Networks, this project combines **graph representation learning** with **gradient boosting**.

The workflow consists of two stages:

## Stage 1 — Graph Representation Learning

A hybrid **GraphSAGE + Graph Attention Network (GAT)** learns structural representations for every transaction.

The network captures:

- Local transaction behaviour
- Multi-hop neighborhood information
- Network topology
- Structural transaction patterns

Each transaction is transformed into a **256-dimensional graph embedding** summarizing its position within the financial network.

---

## Stage 2 — Hybrid Classification

The learned graph embeddings are concatenated with the original **165 engineered transaction features**.

```text
165 Engineered Features
             │
             ▼

256 Graph Embeddings
             │
             ▼

421-D Feature Vector
             │
             ▼

LightGBM Classifier
```

The LightGBM model combines structural information from the graph with transaction-level attributes to produce the final AML risk probability.

This hybrid architecture leverages the strengths of both Graph Neural Networks and gradient boosting, improving predictive performance while maintaining compatibility with SHAP explainability.

---

#  Training Workflow

```text
Bitcoin Transactions
          │
          ▼

Temporal Graph Split
          │
          ▼

Graph Construction
          │
          ▼

Hybrid GraphSAGE + GAT
          │
          ▼

256-D Embeddings
          │
          ▼

Merge with Local Features
          │
          ▼

421-D Hybrid Dataset
          │
          ▼

LightGBM Training
          │
          ▼

Leak-Proof Evaluation
```

---

#  Model Performance

The final model was evaluated on completely unseen transactions from **Timesteps 35–49**.

## Final Results

| Metric | Score |
|---------|------:|
| **PR-AUC** | **0.7158** |
| **Illicit Precision** | **0.87** |
| **Illicit Recall** | **0.58** |
| **Illicit F1-Score** | **0.6932** |
| **Overall Accuracy** | **0.97** |

---

## Classification Report

| Class | Precision | Recall | F1-Score |
|--------|----------:|-------:|----------:|
| **Licit** | **0.97** | **0.99** | **0.98** |
| **Illicit** | **0.87** | **0.58** | **0.69** |

---

## Why PR-AUC?

Money laundering detection is a **highly imbalanced classification problem**, where illicit transactions represent only a small fraction of all transactions.

For this reason, **PR-AUC** and **Illicit-Class F1** provide a more meaningful assessment than overall accuracy alone, as they better capture the model's ability to identify suspicious activity while minimizing false positives.

---

#  Key Design Decisions

The project emphasizes production realism rather than benchmark optimization.

Key design choices include:

- ✅ Strict chronological evaluation
- ✅ Leak-proof temporal training
- ✅ Hybrid Graph + Gradient Boosting architecture
- ✅ Graph embeddings learned before downstream classification
- ✅ Explainable predictions using SHAP
- ✅ Production-ready inference pipeline

# Explainable AI

Financial crime investigations require more than accurate predictions—they require **transparent, interpretable decisions**.

This platform integrates **SHAP (SHapley Additive Explanations)** to explain every prediction generated by the Hybrid Graph Neural Network + LightGBM pipeline.

Rather than treating the model as a black box, analysts can understand how graph-derived information and transaction-level features contributed to the final AML risk score.

---

## Individual Transaction Analysis

For every transaction, the dashboard displays:

- AML Risk Probability
- Risk Recommendation
- Prediction Confidence
- SHAP Waterfall Visualization

Each prediction is decomposed into three interpretable components:

| Component | Description |
|-----------|-------------|
| **Graph Embeddings** | Structural information learned from the Hybrid Graph Neural Network |
| **Local Features** | Original transaction attributes |
| **One-Hop Features** | Aggregated neighborhood information |

These contributions are accumulated from the model's baseline prediction to the final Probability of Illicit Activity, providing a transparent explanation of the decision. :contentReference[oaicite:1]{index=1}

<img width="1823" height="908" alt="Screenshot 2026-07-10 at 6 28 48 PM" src="https://github.com/user-attachments/assets/c64c8af2-bece-44af-b540-e15eaaf75387" />


---

#  Model Performance Monitoring

Monitoring production ML systems is as important as training them.

The platform integrates **Evidently AI** to perform on-demand performance validation on unseen transactions.

Unlike static evaluation performed during training, this monitoring layer enables analysts to continuously verify model quality after deployment.

---

## Evidently AI Audit

The monitoring service performs the following workflow:

1. Randomly samples **5,000 transactions** from the unseen test period.
2. Generates fresh predictions using the deployed LightGBM model.
3. Compares predictions with ground-truth labels.
4. Produces an interactive HTML performance report.

The report includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss
- Confusion Matrix
- Classification Metrics

This functionality is exposed through the FastAPI **/performance** endpoint. :contentReference[oaicite:2]{index=2}

<img width="1874" height="859" alt="Screenshot 2026-07-10 at 6 15 02 PM" src="https://github.com/user-attachments/assets/3b73757e-97a0-4d0f-ae5e-455cb46c47b3" />

---

# Frontend Application

The analyst interface is built using **Streamlit** and provides a clean workflow for investigating suspicious cryptocurrency transactions. :contentReference[oaicite:3]{index=3}

## Features

- Interactive Landing Page
- Transaction Risk Assessment
- SHAP Explainability
- Performance Monitoring
- Responsive Analyst Dashboard
- Modern SaaS-Inspired Interface

---

## Prediction Workflow

```text
Transaction ID
        │
        ▼

Prediction Request
        │
        ▼

FastAPI Backend
        │
        ▼

Hybrid GNN + LightGBM
        │
        ▼

SHAP Computation
        │
        ▼

Interactive Dashboard
```

---

#  FastAPI Backend

The backend is implemented using **FastAPI** and exposes REST endpoints for inference and model monitoring. :contentReference[oaicite:4]{index=4}

## Available Endpoints

| Endpoint | Purpose |
|----------|---------|
| **/predict** | Predict AML risk and return SHAP contribution values |
| **/performance** | Generate an Evidently AI classification report |

---

## Prediction API

For each transaction, the API returns:

- Probability of Illicit Activity
- Baseline Probability
- Graph Embedding Contribution
- Local Feature Contribution
- One-Hop Feature Contribution

This lightweight API design enables the frontend to generate interactive visualizations while keeping inference logic centralized. :contentReference[oaicite:5]{index=5}

---

#  AWS Deployment

The platform is deployed using a containerized AWS infrastructure designed for reproducible inference.

## Infrastructure

- Amazon EC2
- Amazon ECR
- Docker
- FastAPI
- Streamlit

---

## Deployment Workflow

```text
GitHub
    │
    ▼

GitHub Actions
    │
    ▼

Docker Image
    │
    ▼

Amazon ECR
    │
    ▼

AWS EC2
    │
    ▼

FastAPI + Streamlit
```

---

#  Containerization

The application is deployed as isolated Docker services to ensure portability and reproducibility.

| Container | Responsibility |
|-----------|----------------|
| **Frontend** | Streamlit Analyst Portal |
| **Backend** | FastAPI REST API |
| **Inference Engine** | Hybrid LightGBM Model |
| **Monitoring** | Evidently AI Performance Audit |

### Benefits

- Reproducible deployments
- Simplified infrastructure
- Environment isolation
- Production portability

---

#  End-to-End Production Workflow

```text
GitHub Commit
        │
        ▼

Docker Build
        │
        ▼

Amazon ECR
        │
        ▼

AWS EC2
        │
        ▼

FastAPI Service
        │
        ▼

Streamlit Dashboard
        │
        ▼

Prediction + Explainability
        │
        ▼

Evidently AI Monitoring
```
# Repository Structure

```text
AML_GRAPH_INTELLIGENCE_PLATFORM
│
├── api/
│   ├── main.py
│   ├── req_data/
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
|
├── notebooks/
|
├── README.md
└── requirements.txt
```

---

# Technology Stack

| Category | Technologies |
|-----------|--------------|
| **Programming** | Python |
| **Graph Machine Learning** | PyTorch Geometric, GraphSAGE, Graph Attention Networks (GAT) |
| **Gradient Boosting** | LightGBM |
| **Explainability** | SHAP |
| **Monitoring** | Evidently AI |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Cloud** | AWS EC2, Amazon ECR |
| **Containerization** | Docker |
| **Validation** | Pydantic |

---

# Key Skills Demonstrated

## Graph Machine Learning

- Hybrid Graph Neural Networks
- GraphSAGE
- Graph Attention Networks (GAT)
- Representation Learning
- Node Classification

---

## Financial Crime Analytics

- Anti-Money Laundering (AML)
- Illicit Transaction Detection
- Bitcoin Transaction Networks
- Financial Crime Intelligence
- Transaction Risk Scoring

---

## Machine Learning

- LightGBM
- Hybrid Graph + Tabular Learning
- PR-AUC Optimization
- Imbalanced Classification
- Feature Engineering

---

## Explainable AI

- SHAP Explainability
- Waterfall Visualizations
- Prediction Attribution
- Model Transparency

---

## MLOps & Deployment

- Docker Containerization
- AWS Deployment
- FastAPI
- Streamlit
- GitHub Actions
- Evidently AI Monitoring

---

# Production Features

The platform provides an end-to-end AML workflow including:

- Graph-based transaction intelligence
- Leak-proof chronological inference
- Hybrid Graph Neural Network classification
- Real-time prediction API
- Interactive analyst dashboard
- SHAP-powered explainability
- On demand performance monitoring
- Containerized cloud deployment

---

# Future Enhancements

Potential future improvements include:

- Online graph inference for streaming transactions
- Graph drift monitoring
- Automated model retraining
- Transaction network visualization
- Multi-hop explanation graphs
- Active learning for analyst feedback

---

# References

### Dataset

- Elliptic Bitcoin Transaction Dataset

### Libraries

- PyTorch Geometric
- LightGBM
- SHAP
- Evidently AI
- FastAPI
- Streamlit

---

# Author

**Rageshwer Singh**

**M.Sc. Data Science & Artificial Intelligence (BITS Pilani)**

Specialization: **Banking, Financial Services & Insurance (BFSI)**

### Areas of Interest

- Graph Machine Learning
- Financial Crime Analytics
- Credit Risk Modeling
- Explainable AI
- MLOps
- Machine Learning Engineering

---

# Project Summary

This project demonstrates the complete lifecycle of a production-ready **Graph Machine Learning system** for Anti-Money Laundering.

From **temporal graph construction** and **hybrid Graph Neural Networks** to **explainable AI**, **performance monitoring**, and **cloud deployment**, it showcases the engineering practices required to move graph-based fraud detection models beyond research and into deployable analytical systems.

---

# If you found this project interesting

If you found this repository useful or interesting, consider giving it a **⭐**.

Feedback, suggestions, and contributions are always welcome.
