import re
from sentence_transformers import util

#  Basic QA Metrics

def normalize_answer(s: str) -> str:
    """
    Lowercase, strip punctuation/articles, and fix whitespace.
    Used in EM and F1 calculations.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(pred: str, ground_truth: list[str]) -> int:
    """
    Compute Exact Match (EM) between prediction and a list of ground-truth strings.
    Returns 1 if any normalized ground-truth exactly equals the normalized prediction, else 0.
    """
    prediction_norm = normalize_answer(pred)
    ground_truth_norms = [normalize_answer(gt) for gt in ground_truth]
    match = 1 if prediction_norm in ground_truth_norms else 0

    return match

def f1_score(pred: str, gts: list[str]) -> float:
    """
    Compute the maximum token-level F1 across all ground-truth strings.
    Tokens are formed by splitting on whitespace after normalization.
    """
    def single_f1(p: str, gt: str) -> float:
        p_tokens = normalize_answer(p).split()
        gt_tokens = normalize_answer(gt).split()
        common = set(p_tokens) & set(gt_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(p_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    return max(single_f1(pred, gt) for gt in gts)

#  Retrieval Metrics (pure‐local)

def context_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Recall@k: (# of relevant IDs in top-k) / (# of all relevant IDs).
    - retrieved_ids: ordered list of all retrieved context IDs (length >= k)
    - relevant_ids: set of ground-truth context IDs
    - k: cutoff
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit_count = len(top_k & relevant_ids)
    return hit_count / len(relevant_ids)

def mrr_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    MRR@k (for a single query):
    - If the first relevant ID appears at rank i (1-based) in retrieved_ids[:k], return 1/i.
    - If no relevant ID is in top-k, return 0.
    """
    for idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / idx
    return 0.0

def map_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Mean Average Precision at k.
    Rewards retrieving all relevant IDs early.
    """
    hits, sum_prec = 0, 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_prec += hits / i
    if not relevant_ids:
        return 0.0
    return sum_prec / len(relevant_ids)

# End-to-End Helpers

def end_to_end_exact_match(pred_answer: str, ground_truths: list[str]) -> int:
    return exact_match(pred_answer, ground_truths)

def end_to_end_f1(pred_answer: str, ground_truths: list[str]) -> float:
    return f1_score(pred_answer, ground_truths)