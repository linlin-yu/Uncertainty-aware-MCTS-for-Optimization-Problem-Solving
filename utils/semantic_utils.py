from collections import defaultdict
import numpy as np

def analyze_response_semantics(
    responses,
    question,
    response_logprobs,
    response_entropies,
    check_entailment_function,
):
    """
    Cluster responses semantically and compute cluster statistics with optimized entailment checking.

    Parameters:
    -----------
    responses : list[str]
        List of generated responses to analyze.
    question : str
        The question that all responses are answering.
    response_logprobs : list[float]
        Log-probabilities of each response.
    response_entropies : list[float]
        Entropy values of each response.
    check_entailment_function : callable
        Function with signature check_equivalence(problem_description, formulation_A, formulation_B)
        that returns 1 for equivalence (bidirectional entailment), 0 for no equivalence.
    strict_entailment : bool, optional
        Whether both entailment directions must be strict (default: False).

    Returns:
    --------
    tuple[list[dict], float]
        - List of cluster summaries, each containing:
          'component', 'prob', 'variance_entropy', 'mean_entropy'
        - Overall semantic entropy across clusters
    """

    # Early exit for single response
    if len(responses) == 1:
        return [
            {
                "component": responses[0],
                "prob": 1.0,
                "variance_entropy": 0.0,
                "mean_entropy": response_entropies[0],
            }
        ], 0.0

    # Cache for equivalence results to avoid duplicate checks
    entailment_cache = {}

    def get_equivalence(idx1, idx2):
        """Get cached equivalence result or compute if not cached."""
        # Use sorted indices for consistent caching regardless of order
        cache_key = tuple(sorted([idx1, idx2]))
        if cache_key not in entailment_cache:
            result = check_entailment_function(
                question, responses[idx1], responses[idx2]
            )
            entailment_cache[cache_key] = result
        return entailment_cache[cache_key]

    def are_semantically_equivalent(idx1, idx2):
        """Determine if two responses are semantically equivalent using indices."""
        # Skip if identical responses
        if responses[idx1] == responses[idx2]:
            return True

        return get_equivalence(idx1, idx2) == 1

    # Optimized clustering using Union-Find approach
    parent = list(range(len(responses)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Pre-filter: group identical responses first (no entailment check needed)
    text_to_indices = defaultdict(list)
    for i, response in enumerate(responses):
        text_to_indices[response].append(i)

    # Union identical responses
    for indices in text_to_indices.values():
        if len(indices) > 1:
            for i in range(1, len(indices)):
                union(indices[0], indices[i])

    # Only check entailment between different text groups
    unique_texts = list(text_to_indices.keys())
    representative_indices = [text_to_indices[text][0] for text in unique_texts]

    # Optimize comparison order: compare high-probability responses first
    # (they're more likely to form large clusters)
    representative_indices.sort(key=lambda idx: response_logprobs[idx], reverse=True)

    # Check equivalence only between representatives of different text groups
    for i, idx1 in enumerate(representative_indices):
        for idx2 in representative_indices[i + 1 :]:
            if find(idx1) != find(idx2) and are_semantically_equivalent(idx1, idx2):
                union(idx1, idx2)

    # Build final clusters
    clusters = defaultdict(list)
    for i in range(len(responses)):
        root = find(i)
        clusters[root].append(i)

    # Compute cluster statistics
    total_responses = len(responses)
    cluster_summaries = []
    overall_semantic_entropy = 0.0

    for cluster_indices in clusters.values():
        cluster_size = len(cluster_indices)
        cluster_prob = cluster_size / total_responses

        # Find best response (highest logprob) in cluster
        best_idx = max(cluster_indices, key=lambda idx: response_logprobs[idx])
        best_response = responses[best_idx]

        # Compute entropy statistics for this cluster
        cluster_entropies = [response_entropies[idx] for idx in cluster_indices]
        mean_entropy = np.mean(cluster_entropies)
        variance_entropy = (
            np.var(cluster_entropies) if len(cluster_entropies) > 1 else 0.0
        )

        # Accumulate semantic entropy: -p * log(p)
        overall_semantic_entropy += -cluster_prob * np.log(cluster_prob + 1e-12)

        cluster_summaries.append(
            {
                "component": best_response,
                "prob": cluster_prob,
                "variance_entropy": variance_entropy,
                "mean_entropy": mean_entropy,
            }
        )

    return cluster_summaries, overall_semantic_entropy
