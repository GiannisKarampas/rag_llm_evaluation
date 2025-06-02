import json
import time
from pathlib import Path
import pandas as pd

from embeddings.embedding_model import EmbeddingModel
from llm_api.lmstudio_client import LlmStudioClient
from vector_db.chroma_client import ChromaVectorDB
from evaluation.metrics import *


def load_data(path: Path) -> list[dict]:
    """
    Expect each item in JSON to have keys:
       - "question": str
       - "context_id": str           # unique ID for the ground-truth context
       - "context": str              # ground-truth passage text
       - "answer": list[str]         # list of acceptable ground truths
    """
    print(f"Loading data from {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if isinstance(item["answer"], str):
            item["answer"] = [item["answer"]]
        if isinstance(item.get("context_ids", []), str):
            item["context_ids"] = [item["context_ids"]]
        if "contexts" not in item:
            item["contexts"] = []
    return data

def main():
    print("Loading data...")
    dataset_path = Path("data/squad_sample.json")
    data = load_data(dataset_path)
    print("Loaded", len(data), "examples")

    embedding_model = EmbeddingModel()
    chroma_db = ChromaVectorDB()
    llm = LlmStudioClient()

    # Index all ground-truth contexts into ChromaDB with their IDs
    context_id_to_text = {}
    for item in data:
        for cid, ctx in zip(item["context_ids"], item.get("contexts", [])):
            context_id_to_text[cid] = ctx
    context_ids = list(context_id_to_text.keys())
    contexts = list(context_id_to_text.values())
    metadata = [{"source": cid} for cid in context_ids]
    chroma_db.add_texts(contexts, metadata, ids=context_ids)

    results = []
    for i, item in enumerate(data):
        q = item["question"]
        ground_truth_ids = item["context_ids"]
        ground_truth_contexts = [context_id_to_text[cid] for cid in ground_truth_ids]
        ground_truth_answer = item["answer"]

        start = time.time()
        retrieved = chroma_db.query(q, embedding_model.get_embedding, top_k=5)
        retrieval_latency = (time.time() - start) * 1000
        retrieved_ids = retrieved["ids"][0]
        retrieved_contexts = retrieved["documents"][0]

        # Compute Retrieval metrics
        ctx_recall_5 = context_recall_at_k(retrieved_ids, set(ground_truth_ids), k=5)
        mrr_5 = mrr_at_k(retrieved_ids, set(ground_truth_ids), k=5)
        map_5 = map_at_k(retrieved_ids, set(ground_truth_ids), k=5)

        # --- Generation step ---
        prompt = (
                "You are a knowledgeable assistant. Use the following evidence passages to answer the question as specifically as possible.\n"
                + "\n\n".join(
            [f"Passage {i + 1}:\n{ctx}" for i, ctx in enumerate(retrieved_contexts)]
        )
                + f"\n\nQuestion: {q}\nAnswer:"
        )
        start_gen = time.time()
        pred_answer = llm.ask(prompt)
        generation_latency = time.time() - start_gen

        # Compute Generation metrics
        gen_em = exact_match(pred_answer, ground_truth_answer)
        gen_f1 = f1_score(pred_answer, ground_truth_answer)

        # End-to-End metrics (same as generation EM/F1)
        e2e_em = gen_em
        e2e_f1 = gen_f1


        # Append to results
        results.append({
            "question": q,
            "ground_truth_answers": ground_truth_answer,
            "retrieved_ids": retrieved_ids,
            "ctx_recall@5": ctx_recall_5,
            "mrr@5": mrr_5,
            "map@5": map_5,
            "pred_answer": pred_answer,
            "gen_em": gen_em,
            "gen_f1": gen_f1,
            "e2e_em": e2e_em,
            "e2e_f1": e2e_f1,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
        })

        # (Optional) Per-question console output
        print(f"Q: {q}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Context Recall@5: {ctx_recall_5:.2f}, MRR@5: {mrr_5:.2f}, MAP@5: {map_5:.2f}")
        print(f"Pred Answer: {pred_answer}")
        print(f"Gen EM: {gen_em}, Gen F1: {gen_f1:.2f}")
        print(f"E2E EM: {e2e_em}, E2E F1: {e2e_f1:.2f}")
        print("-" * 50)

        # 3. Save all results to JSON for downstream analysis/dashboard
        pd.DataFrame(results).to_json("eval_results.json", orient="records")
        print("Finished evaluation. Results saved to eval_results.json")

if __name__ == "__main__":
    main()