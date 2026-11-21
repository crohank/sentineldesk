"""Copy the managed collection's vectors for exact search during quota exhaustion."""
from pathlib import Path
import numpy as np
import pandas as pd
from sentineldesk.agent_retrieval import AgentRetrieval


def main():
    store = AgentRetrieval()
    ids, vectors = [], []
    token = ""
    while True:
        page = store.post(store.objects + ":query", {"pageSize": 250, "pageToken": token,
                          "outputFields": {"dataFields": ["ticket_id"], "vectorFields": ["embedding"]}})
        for obj in page.get("dataObjects", []):
            ids.append(obj["data"]["ticket_id"])
            vectors.append(np.asarray(obj["vectors"]["embedding"]["dense"]["values"], dtype=np.float32))
        token = page.get("nextPageToken")
        if not token:
            break
    root = Path("data/processed")
    rows = pd.read_parquet(root/"kb_train.parquet", columns=["ticket_id","answer"]).merge(
        pd.read_parquet(root/"kb_partitions.parquet"),validate="one_to_one").query("partition=='fit'")
    assert set(ids) == set(rows.loc[rows.answer.fillna("").str.strip().ne(""),"ticket_id"]) and len(ids)==18280
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    assert matrix.shape==(18280,768) and np.isfinite(matrix).all() and (norms>0).all()
    root = Path("artifacts/fast_path")
    np.save(root/"vectors.npy",matrix/norms)
    np.save(root/"ticket_ids.npy",np.asarray(ids))
    print("Verified snapshot",matrix.shape,flush=True)


if __name__=="__main__":
    main()
