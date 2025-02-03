from cryptography.hazmat.primitives import hashes
from typing import List
import numpy as np


def calculate_gradient_hash(gradients: List[np.ndarray]) -> bytes:
    """Calculate deterministic hash of model gradients."""
    hasher = hashes.Hash(hashes.SHA256())

    for grad in gradients:
        # Convert gradient values to bytes in a deterministic way
        grad_bytes = grad.tobytes()
        hasher.update(grad_bytes)

    return hasher.finalize()


class MLProofOfWork:
    def __init__(self):
        self.gradient_hash_size = 32
        self.proof_difficulty = 0.1  # Adjust based on desired verification strictness

    def verify_gradient_continuity(
        self, prev_gradients: List[np.ndarray], current_gradients: List[np.ndarray]
    ) -> bool:
        """Verify that gradients change follows expected training patterns."""
        if not prev_gradients or not current_gradients:
            return True

        total_diff = 0
        for prev, curr in zip(prev_gradients, current_gradients):
            # Calculate cosine similarity between gradient updates
            similarity = np.dot(prev.flatten(), curr.flatten()) / (
                np.linalg.norm(prev) * np.linalg.norm(curr)
            )
            total_diff += abs(1 - similarity)

        avg_diff = total_diff / len(prev_gradients)
        return avg_diff <= self.proof_difficulty

    def verify_loss_trajectory(self, loss_history: List[float]) -> bool:
        """Verify that loss values follow expected training patterns."""
        if len(loss_history) < 3:
            return True

        # Check if loss generally decreases over time (allowing for some fluctuations)
        window_size = 3
        for i in range(len(loss_history) - window_size):
            window = loss_history[i : i + window_size]
            if np.mean(window[:-1]) < np.mean(window[1:]) * 0.9:  # Allow 10% increase
                return False

        return True
