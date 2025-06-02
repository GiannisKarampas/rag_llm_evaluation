# RAG-LLM Multi-Hop QA Evaluation

This repository provides a complete pipeline to evaluate Retrieval-Augmented Generation (RAG) QA systems on **multi-hop, multi-context** questions. It combines local embeddings (SentenceTransformers), a vector database (ChromaDB), a local LLM (LM Studio)for advanced context-recall metrics and claim coverage.

---

## 📦 Project Structure

````
.
├── data/
│└── squad_sample.json
├── embeddings/
│   └── embedding_model.py
├── vector_db/
│   └── chroma_client.py
├── llm_api/
│   └── lmstudio_client.py
├── evaluation/
│   └── metrics.py
├── eval.py
├── ui_dashboard.py
└── README.md
````
---
## 🚀 Features
- **Multi-hop QA**: Each question may require two or more context passages for the correct answer.
- **Retrieval Metrics**:  
  - **Recall@k** – proportion of gold contexts retrieved in top-k.
  - **MRR@k** – reciprocal rank of the first correct context.
  - **MAP@k** – Mean Average Precision.
- **Generation Metrics**:
  - **Exact Match (EM)**, **F1 Score**: LLM output vs. ground-truth answer(s).
  - **End-to-End EM/F1**: Pipeline success (retrieval + generation).
- **Latency Logging**: Retrieval and generation times.
- **Streamlit Dashboard**: KPIs, filtering, distributions, error analysis, and per-question drill-down.

---

## ⚙️ Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/GiannisKarampas/rag_llm_evaluation.git
    cd RAG_LLM_evaluation
    ```

2. **(Recommended) Create a virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

4. **Start LM Studio**  
   - Ensure it’s running at `http://localhost:1234/v1/chat/completions`  
   - Load your preferred LLM (e.g., Llama-3, Mistral, etc.)

---

## 📂 Dataset Format

Each entry in `data/squad_sample.json`:

```json
{
  "question": "Which city is the birthplace of the spouse of the artist who painted The Starry Night?",
  "context_ids": ["c101","c214"],
  "contexts": [
    "Vincent van Gogh painted The Starry Night. His brother Theo van Gogh was married to Johanna Bonger.",
    "Johanna Bonger was born in Amsterdam in 1862."
  ],
  "answer": ["Amsterdam"]
}
```

---

## 🏃‍♂️ Usage
1. **Run the evaluation script**
```bash
 python eval.py
```
2. **Start the Streamlit dashboard**
```bash
 streamlit run ui_dashboard.py
```
Visualizes metrics, distributions, allows filtering and question-level error analysis.

---
## 📊 Metrics Explained
- **Recall@k**: Fraction of required ground-truth context IDs in top-k retrieved.
- **MRR@k**: Inverse of the rank of the first ground-truth context retrieved.
- **MAP@k**: How well all relevant contexts are retrieved, with higher weight for early hits.
- **EM/F1**: Exact/partial match between LLM output and references.
- **End-to-End EM/F1**: Measures final pipeline accuracy.
- **Latencies**: Time to retrieve passages and generate answers.

---
## 📈 Dashboard Features
- At-a-glance KPIs
- Interactive filters (Recall, MRR, F1, error-only)
- Metric distributions (bar, histogram, boxplot)
- Question drill-down (view full evidence, answer, ground truth, and metrics)

---
## 📜 License & Credits
- License: MIT
- Thanks to:
  - SentenceTransformers (https://www.sbert.net/)
  - ChromaDB (https://www.trychroma.com/)
  - LM Studio (https://lmstudio.ai/)