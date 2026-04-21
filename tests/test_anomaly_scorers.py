import numpy as np

from src.anomaly.scorers import KNNScorer, MahalanobisScorer


def test_mahalanobis_scorer_smoke():
    train = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    scorer = MahalanobisScorer.fit(train, regularization=1e-3)
    scores = scorer.score(np.array([[0.05, 0.05], [3.0, 3.0]]))
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_knn_scorer_smoke():
    train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    scorer = KNNScorer(train_embeddings=train, k=2)
    scores = scorer.score(np.array([[0.0, 0.0], [2.0, 2.0]]))
    assert scores.shape == (2,)
    assert scores[1] > scores[0]

