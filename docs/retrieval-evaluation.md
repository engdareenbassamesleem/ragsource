# Retrieval evaluation (offline baseline)

This module evaluates ranked **chunk IDs** against labeled relevant chunk IDs. It does not evaluate answer factuality or hallucinations.

Metrics at cutoff K:
- Precision@K = relevant retrieved IDs among first K / K.
- Recall@K = relevant retrieved IDs among first K / number of relevant IDs.
- Reciprocal Rank@K = 1 / rank of the first relevant result in the first K, or 0 if absent.

The evaluation is deterministic, requires no API key, and does not download a model.

Example:

```python
from ragsource.evaluation import evaluate_ranking
score = evaluate_ranking(["c1", "c2", "c3"], {"c2", "c4"}, k=3)
print(score)
```

Use `mean_metrics` for macro averages across queries. Empty ground truth has explicitly defined recall 0; do **not** interpret that as a retrieval error or as evidence of good performance. Repeated retrieved IDs are rejected. Precision uses K in the denominator even when fewer than K items are returned.

Before reporting benchmark numbers, build a manually reviewed labeled query set and freeze corpus, chunking, embedder, dataset split, and K. Report source, dataset size, sample composition and failure cases. No benchmark accuracy claim is made by this module.
