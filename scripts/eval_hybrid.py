"""Select a per-head TF-IDF confidence cutoff, then score the disjoint validation half."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
from scripts.calibrate_managed import CACHE, ROOT
from scripts.score import detect_language
from sentineldesk.triage import TARGETS, TriagePredictor
from sentineldesk.safety.pii import mask_pii

def main():
    data = ROOT / "data/processed"
    rows = pd.read_parquet(data/"kb_train.parquet", columns=["ticket_id","subject","body",*TARGETS]).merge(
        pd.read_parquet(data/"kb_partitions.parquet"),validate="one_to_one").query("partition=='val'").sample(300, random_state=42).reset_index(drop=True)
    with sqlite3.connect(CACHE) as db:
        raw = {k:json.loads(v) for k,v in db.execute("select id,payload from predictions")}
    for field in ("subject","body"):
        rows[field] = rows[field].fillna("").map(lambda v: mask_pii(v).text)
    bundle = ROOT/"artifacts/fast_path"
    model = TriagePredictor.load(bundle)
    tf = model.predict_batch(rows)
    methods = {name:pd.DataFrame(index=rows.index) for name in ("tfidf","gemini","blend_50_50","hybrid")}
    cutoffs = {}
    for h in TARGETS:
        g = pd.Series([raw[k][h]["label"] for k in rows.ticket_id])
        t = tf[f"{h}_pred"]
        conf = tf[f"{h}_conf"]
        options = [1.01, .9, .8, .7, 0.]
        def selection_score(cutoff):
            selected = g.where(conf < cutoff, t)
            return f1_score(rows[h][:150], selected[:150],average="macro",zero_division=0)
        cutoff = max(options,key=selection_score)
        cutoffs[h] = cutoff
        methods["tfidf"][h], methods["gemini"][h] = t,g
        methods["hybrid"][h] = g.where(conf < cutoff,t)
        labels = list(model.heads[h].classes_)
        blend = pd.DataFrame({
            label: (tf[f"p_{h}_{label}"] + pd.Series([raw[k][h]["probabilities"][label] for k in rows.ticket_id])) / 2
            for label in labels
        })
        methods["blend_50_50"][h] = blend.idxmax(axis=1)
    langs = rows.body.iloc[150:].map(detect_language)
    report = {"selection_rows":150,"validation_rows":150,"cutoffs":cutoffs,"results":{}}
    for lang,idx in [("all",langs.index),("en",langs[langs=="en"].index),("de",langs[langs=="de"].index)]:
        actual = rows.loc[idx]
        group = {"tickets":len(idx)}
        high = actual.type.eq("Incident") & actual.priority.eq("high")
        for name,df in methods.items():
            pred = df.loc[idx]
            group[name] = {h:round(f1_score(actual[h],pred[h],average="macro",zero_division=0),4) for h in TARGETS}
            group[name]["high_incidents_caught"] = int((high & pred.type.eq("Incident") & pred.priority.eq("high")).sum())
            group[name]["actual_high_incidents"] = int(high.sum())
        report["results"][lang]=group
    (bundle/"selection.json").write_text(json.dumps(cutoffs,indent=2))
    (ROOT/"reports/llm/hybrid_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    main()
