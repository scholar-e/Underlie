import numpy as np

def compare_tribev2_predictions(preds_a, preds_b):
    """
    Compare two TRIBE v2 prediction outputs:
    one from the AI-created sample and one from the test sample.
    """

    a = np.asarray(preds_a, dtype=float)
    b = np.asarray(preds_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    # Mean Squared Error, Mean Absolute Error, Correlation Coefficient, Cosine Similarity
    # Mean Squared Error: The smaller the MSE is, the more similar the two outputs are.
    mse = np.mean((a - b) ** 2)
    # Mean Absolute Error： The smaller the MAE is, the more similar the two outputs are.
    mae = np.mean(np.abs(a - b))

    a_flat = a.flatten()
    b_flat = b.flatten()
    # Correlation Coefficient: The closer the correlation coefficient is to 1, the more similar the two outputs are.
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(a_flat, b_flat)[0, 1]
    # Cosine Similarity: The closer the cosine similarity is to 1, the more similar the two outputs are.
    cosine_similarity = np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    )

    return {
        "mse": float(mse),
        "mae": float(mae),
        "correlation": float(correlation),
        "cosine_similarity": float(cosine_similarity),
    }


if __name__ == "__main__":
    # fake TRIBE v2 outputs for testing
    preds1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    preds2 = np.array([[0.1, 0.25, 0.28], [0.42, 0.48, 0.61]])

    result = compare_tribev2_predictions(preds1, preds2)
    print(result)