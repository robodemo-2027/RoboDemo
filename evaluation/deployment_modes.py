"""Action selection only; callers must update policy memory on every real step."""

from collections import deque

import numpy as np


class ActionSelector:
    def __init__(self, mode):
        if mode not in ("replan1", "chunk4", "temporal4"):
            raise ValueError(mode)
        self.mode = mode
        self.history = deque(maxlen=4)
        self.last_step = -1

    def select(self, step, prediction):
        prediction = np.asarray(prediction, dtype=np.float32)
        if step != self.last_step + 1:
            raise ValueError("Every control step must be accounted for")
        if prediction.shape != (30, 16) or not np.isfinite(prediction).all():
            raise ValueError("Expected a finite 30-by-16 predicted action chunk")
        self.last_step = step
        if self.mode == "replan1":
            return prediction[0].copy()
        if step % 4 == 0:
            self.history.append((step, prediction.copy()))
        if self.mode == "chunk4":
            origin, chunk = self.history[-1]
            return chunk[step - origin].copy()
        actions, weights = [], []
        for age, (origin, chunk) in enumerate(reversed(self.history)):
            offset = step - origin
            if 0 <= offset < len(chunk):
                actions.append(chunk[offset])
                weights.append(np.exp(-0.1 * age))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        return np.sum(np.asarray(actions) * weights[:, None], axis=0).astype(np.float32)
