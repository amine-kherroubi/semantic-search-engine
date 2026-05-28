# Mini-Project: Semantic Search Engine with a Vector Database

## Overview

Design and implement an intelligent document retrieval system that finds documents by **meaning** rather than exact keyword matching. The system accepts natural-language queries and returns the most semantically relevant documents from a vector database.

---

## Learning Objectives

By the end of this project, students will be able to:

- Explain how vector databases work and when to use them
- Generate and store text embeddings using a pre-trained model
- Design a full semantic search pipeline
- Implement similarity-based retrieval
- Compare semantic search against traditional lexical methods (TF-IDF / keyword)
- Produce a structured technical report

---

## Technical Requirements

| Component        | Technology             |
| ---------------- | ---------------------- |
| Language         | Python                 |
| Vector database  | PostgreSQL + pgvector  |
| NLP / Embeddings | Sentence-Transformers  |
| Dataset source   | Kaggle or Hugging Face |

All technologies must be **open-source**.

---

## Dataset

Choose any public dataset containing **at least 1,000 text documents**. Suggested options:

- Scientific article abstracts (arXiv)
- News articles
- Technical FAQs
- Short legal documents
- Customer reviews
- Medical texts: clinical case notes (MIMIC-III), disease descriptions, or datasets from PhysioNet / Hugging Face

---

## Deliverables & Phases

### Phase 1 - Analysis & Design
- Problem statement and dataset justification
- System architecture diagram
- Choice of vectorization method with rationale
- Database schema

### Phase 2 - Vector Database Implementation
- Text preprocessing pipeline
- Embedding generation
- Vector storage and indexing in pgvector

### Phase 3 - Semantic Search
- Query vectorization at runtime
- Similarity computation (cosine or L2 distance)
- Top-*k* document retrieval
- Side-by-side comparison with a classical search method (TF-IDF or keyword)

### Phase 4 - Validation & Critical Analysis
- Functional test cases
- Performance analysis
- Discussion of system limitations
- Proposed improvements

---

## Final Submission

Submit a complete project folder containing:

1. **Documented source code**
2. **SQL scripts** or pgvector configuration files
3. **Technical report (10-15 pages)** structured as follows:
   - Introduction
   - State of the art (vector databases)
   - Methodology
   - Implementation details
   - Results
   - Critical discussion
   - Conclusion

---

## Bonus (Optional)

- Multilingual query support
- Comparative evaluation of multiple embedding models
