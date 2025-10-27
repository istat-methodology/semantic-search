from typing import List, Any, Dict
from semantic_search.data import SearchOutput
from semantic_search.data import EvaluationEntry, EvaluationOutput


def _get_rank(y_t: Any, y_p: List[Any]) -> int:
    r"""Get rank of true label in predicted labels list. Returns 0 if not found."""
    return y_p.index(y_t) + 1 if y_t in y_p else 0


def run_evaluation(
    search_outs: List[SearchOutput],
    label_key: Any,
    y_true: List[Any],
) -> EvaluationOutput:
    r"""Run evaluation given search outputs and true labels.

    Args:
        search_outs (List[SearchOutput]): List of search outputs.
        label_key (Any): Key to extract label from metadata.
        y_true (List[Any]): List of true labels for each query.

    Returns:
        EvaluationOutput: Evaluation results containing entries for each query.
    """
    y_pred = [[res.metadata[label_key] for res in out.results] for out in search_outs]
    scores = [[res.score for res in out.results] for out in search_outs]

    ranks = [_get_rank(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)]

    eval_results = []
    for y_t, y_p, s_p, rank in zip(y_true, y_pred, scores, ranks):
        is_correct = [int(y_t == y) for y in y_p]
        eval_results.append(
            EvaluationEntry(
                y_true=y_t, y_pred=y_p, scores=s_p, is_correct=is_correct, rank=rank
            )
        )

    return EvaluationOutput(results=eval_results)


def get_metrics(
    eval_results: EvaluationOutput,
    ks: List[int] | None = [1, 3, 5],
    score_threshold: float | None = None,  # To be implemented
    verbose: bool = True,
):
    r"""Compute evaluation metrics from evaluation results.

    Args:
        eval_results (EvaluationOutput): Evaluation results.
        ks (List[int] | None): List of k values for accuracy@k.
        score_threshold (float | None): Score threshold for filtering (to be implemented).
        verbose (bool): Whether to print the results.
    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    ranks = [result.rank for result in eval_results]

    hits_at_ks = {}
    for k in ks:
        hits_at_ks[k] = [int((rank <= k) & (rank != 0)) for rank in ranks]

    results = {}
    for k, hits in hits_at_ks.items():
        accuracy = sum(hits) / len(hits)
        print(f"Accuracy@{k}: {accuracy:.4f}") if verbose else None
        results[f"accuracy@{k}"] = accuracy

    return results
