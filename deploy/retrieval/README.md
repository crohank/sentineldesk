# SentinelDesk managed retrieval

The collection `tickets-fit-ge2-v1` is in project `sentineldesk-ajin-ze3gmb`, region `us-central1`. Its 768-dimensional field contains Gemini Embedding 2 document vectors. Documents contain masked subjects and historical answers. Search embeds the incoming masked text with the query prompt and returns five precedents with cosine similarities. The default search uses exact nearest neighbors.

From the repository root, install the project and the additional runtime dependencies with `python -m pip install -e . -r deploy/retrieval/requirements.txt`. The default credential source is Google Application Default Credentials. For local development with an existing gcloud login, set `SENTINELDESK_AUTH=gcloud`.

To resume or verify ingestion, run `python -m scripts.build_agent_retrieval data/processed` with D1’s canonical kb_train and kb_partitions Parquet files. The builder compares remote IDs with the 18,280 resolved fit ticket IDs. It skips already stored tickets on a rerun. Use a new collection version for changed text, model, dimensions or masking.

Run the console and API with `SENTINELDESK_MODE=managed python -m sentineldesk.api`. Set `SENTINELDESK_DATA_DIR` if the canonical Parquet files are outside `data/processed`. The app runs one retrieval request and uses the returned IDs to select the examples for Gemini 3.8 Flash. Rochan’s safety gate routes outages and sensitive requests. For eligible tickets the app generates a cited draft, validates it against the same precedents, and shows it to the operator.

The managed app drafts replies from reviewed templates and validates each draft before showing it to the operator.

Gemini probabilities are uncalibrated. The gate uses the predicted labels and provisional confidence and similarity floors. The probability-pair outage rule applies only when the predictions have no explicit uncalibrated flag. Validate the floors and calibrate the classifier before relying on these settings.
