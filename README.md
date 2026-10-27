# Credence: The EchoShield 🛡️
### *An Evidence-Based Fake News Verification & Explainability System*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credence-echoshield.streamlit.app)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DistilBERT%20Baseline-yellow)](https://huggingface.co/Shrey-2310/credence-distilbert-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

Traditional fake news detection systems evaluate syntax, vocabulary, and stylistic markers alone, leaving them blind to emerging factual distortions, satire, or deceptive claims framed in neutral journalistic prose. 

**Credence (The EchoShield)** addresses this limitation by moving beyond stylistic guessing into **evidence-based factual verification**. Rather than returning an uninterpretable binary label ("Fake" vs "Real"), Credence:
1. **Isolates concrete factual claims** from raw text or article URLs.
2. **Retrieves real-time external evidence** across verified web/news sources.
3. **Performs Natural Language Inference (NLI)** to determine whether retrieved evidence supports or contradicts the statements.
4. **Fuses contextual stylistic indicators with external evidence** to output a calibrated confidence score and a side-by-side explainable evidence audit.

---

## 🏗️ System Architecture

```text
[ Article URL or Raw Text ]
           │
           ▼
[ Automated Ingestion & Cleaning ] ── (Trafilatura & BeautifulSoup Boilerplate Stripping)
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
[ Baseline Text Classification ]         [ Factual Claim Isolation ]
   (DistilBERT Fine-Tuned)                           │
           │                                         ▼
           │                             [ Live Evidence Retrieval ]
           │                                 (DuckDuckGo News)
           │                                         │
           │                                         ▼
           │                           [ Semantic Evidence Reranking ]
           │                               (Sentence Transformers)
           │                                         │
           │                                         ▼
           │                          [ Cross-Encoder Stance Check ]
           │                               (DeBERTa-v3 NLI)
           │                                         │
           └───────────────────┬─────────────────────┘
                               ▼
                    [ Multi-Source Score Fusion ]
                               │
                               ▼
               [ Explainable Verification Report ]
