# SentinelDesk managed retrieval

The collection `tickets-fit-ge2-v1` is in project `sentineldesk-ajin-ze3gmb`, region `us-central1`. Its 768-dimensional field contains Gemini Embedding 2 document vectors. Documents contain masked subjects and historical answers. Search embeds the incoming masked text with the query prompt and returns five precedents with cosine similarities. The default search uses exact nearest neighbors.

From the repository root, install the project and the additional runtime dependencies with `python -m pip install -e . -r deploy/retrieval/requirements.txt`. The default credential source is Google Application Default Credentials. For local development with an existing gcloud login, set `SENTINELDESK_AUTH=gcloud`.

To resume or verify ingestion, run `python -m scripts.build_agent_retrieval data/processed` with D1’s canonical kb_train and kb_partitions Parquet files. The builder compares remote IDs with the 18,280 resolved fit ticket IDs. It skips already stored tickets on a rerun. Use a new collection version for changed text, model, dimensions or masking.

Run the console and API with `SENTINELDESK_MODE=managed python -m sentineldesk.api`. Set `SENTINELDESK_DATA_DIR` if the canonical Parquet files are outside `data/processed`. The app runs one retrieval request and uses the returned IDs to select the examples for Gemini 3.8 Flash. It returns estimates and precedents for operator review. The estimates are uncalibrated and the five-example managed configuration needs a validation run before its metrics can be compared with D1’s eight-example baseline.
