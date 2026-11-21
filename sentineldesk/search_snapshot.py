"""Exact cosine retrieval over an immutable copy of the remote fit collection."""
import numpy as np
import pandas as pd
from .safety.pii import mask_pii


class SearchSnapshot:
    def __init__(self, bundle, data_dir):
        self.matrix = np.load(bundle / "vectors.npy", mmap_mode="r")
        self.ids = np.load(bundle / "ticket_ids.npy")
        self.rows = pd.read_parquet(data_dir / "kb_train.parquet", columns=["ticket_id","subject","answer"]).set_index("ticket_id")
        if len(self.matrix) != len(self.ids):
            raise ValueError("Incomplete search snapshot")

    def search(self, vector, k):
        query = np.asarray(vector, dtype=np.float32)
        scores = self.matrix @ (query / np.linalg.norm(query))
        indices = np.argsort(-scores, kind="stable")[:k]
        hits = []
        for idx in indices:
            key = str(self.ids[idx])
            row = self.rows.loc[key]
            data = {"ticket_id":key, "subject":mask_pii(row.subject).text, "answer":mask_pii(row.answer).text}
            hits.append({"dataObject":{"data":data},"distance":1-float(scores[idx])})
        return hits
