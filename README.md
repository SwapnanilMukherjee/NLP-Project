# Reddit Topic and Stance Explorer

This project implements the assignment pipeline from `NLP_proj_1.pdf` with Arctic Shift as the data source and `r/gradadmissions` as the default subreddit.

- load Reddit posts and comments into a local SQLite database
- compute aggregate corpus statistics
- extract interpretable topics with labels, keywords, shares, and time series
- separate trending topics from persistent topics
- classify topic-linked comments as supporting or opposing the dominant position
- summarize the main arguments made by each side
- expose everything in an interactive Streamlit application

## Design choices

- `r/gradadmissions` is the default subreddit because it is socially relevant, discussion-heavy, and likely to surface recurring stance conflicts around admissions policy, profiles, and application strategy.
- Arctic Shift replaces PRAW for the main ingestion path. It offers searchable Reddit archives without requiring Reddit API credentials.
- SQLite is the local database because it satisfies the assignment requirement with zero external service setup.
- Topic extraction uses transformer sentence embeddings with MiniBatchKMeans clustering, plus spaCy noun-chunk/entity candidates combined with phrase-filtered c-TF-IDF and embedding reranking, followed by generated labels.
- Trending vs persistent topics are derived from weekly topic-share trajectories using recency lift and slope over time.
- Stance uses a transformer NLI model with explicit agree/disagree hypothesis templates over topic-linked comments and generated topic claims. Summaries and topic labels come from an instruction-tuned causal LM running on the GPU.

## Current NLP pipeline

### Pipeline

1. Data comes from Arctic Shift, gets stored in SQLite, and analysis is run per subreddit, currently `r/gradadmissions`.

2. Each post is converted to one text unit as `title + selftext`, then embedded with `BAAI/bge-base-en-v1.5` on GPU. Those embeddings are clustered with `MiniBatchKMeans` to form topics, and topic keywords are extracted from spaCy noun chunks and named entities, combined with a phrase-filtered c-TF-IDF-style `CountVectorizer` pass and embedding reranking toward each topic centroid.

3. For each topic, `Qwen/Qwen2.5-3B-Instruct` generates:
- a short topic label
- a debate claim/viewpoint for stance analysis
- repaired alternatives if the first claim is weak
- final agreement/disagreement summaries

The code now generates multiple claim candidates, filters out bad ones like emotional summaries or school-specific one-offs, and ranks them with a small NLI probe before picking the final claim.

4. Topic type is then labeled `trending` or `persistent` from weekly topic-share time series using recency lift and slope heuristics, not a neural model.

5. Stance is classified comment-by-comment with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. For each comment, the model checks two hypotheses:
- `The author agrees with the claim: ...`
- `The author disagrees with the claim: ...`

If entailment for one side is strong enough and clearly above the other side, the comment becomes `support` or `oppose`; otherwise it stays `neutral`. Current thresholds are `0.42` strong, `0.30` weak, with a `0.08` label margin.

6. After that:
- a topic gets a dominant stance only if there are enough explicit stance comments and a clear margin; otherwise it is `insufficient_evidence`
- summaries are generated separately for agreement-side and disagreement-side comments
- users are grouped per topic by majority vote over their comment stances

### Exact models

- Topic embeddings: `BAAI/bge-base-en-v1.5`
- Stance / NLI: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- Label / claim generation / summaries: `Qwen/Qwen2.5-3B-Instruct`

### Methods in one line each

- Topic extraction: transformer embeddings + `MiniBatchKMeans`
- Topic keywords: spaCy noun-chunk/entity candidates plus phrase-filtered c-TF-IDF over `CountVectorizer`, then embedding-based reranking
- Topic labels: instruction-tuned LLM generation
- Claim selection: LLM candidate generation + rule-based filtering + small NLI ranking probe
- Trend/persistent labeling: weekly share heuristics
- Stance detection: NLI with agree/disagree hypothesis templates
- Summarization: LLM summaries over top stance-grouped comments
- User grouping: per-topic majority vote over comment stances

## Project layout

- `reddit_insights/`: package source
- `app.py`: Streamlit entrypoint
- `data/`: SQLite database, exports, and analysis artifacts
- `sample_data/`: local JSONL fixtures for smoke testing

## Setup

Always activate the local virtual environment first:

```bash
source bin/activate
```

Optional: copy `.env.example` to `.env` and adjust the Arctic Shift base URL, timeout, or default subreddit.

For the current keyword extraction pipeline, also install the English spaCy model once:

```bash
source bin/activate
python -m spacy download en_core_web_sm
```

The analysis step now expects a CUDA-capable GPU if you keep the default model settings. The tested setup uses a 24 GB GPU with:

- `BAAI/bge-base-en-v1.5` for post embeddings
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for stance/NLI
- `Qwen/Qwen2.5-3B-Instruct` for topic labels, topic viewpoints, and side summaries

## Usage

Initialize the database:

```bash
source bin/activate
python -m reddit_insights.cli init-db
```

Ingest posts and comments from Arctic Shift:

```bash
source bin/activate
python -m reddit_insights.cli ingest \
  --subreddit gradadmissions \
  --days 180 \
  --post-target 15000 \
  --comment-limit 20
```

Run analysis for the configured subreddit:

```bash
source bin/activate
python -m reddit_insights.cli analyze --subreddit gradadmissions --topic-count 10
```

Launch the app:

```bash
source bin/activate
streamlit run app.py
```

Load a local archive instead of calling Arctic Shift live:

```bash
source bin/activate
python -m reddit_insights.cli import-posts --path /path/to/posts.jsonl --subreddit gradadmissions
python -m reddit_insights.cli import-comments --path /path/to/comments.jsonl
python -m reddit_insights.cli analyze --subreddit gradadmissions --topic-count 10
```

## Notes

- Arctic Shift exposes both search endpoints and monthly dumps. For a 15,000+ post target, the API path is convenient for iteration, while the existing JSONL import flow remains useful if you decide to process dumps offline.
- The app expects analysis artifacts to exist. Run `analyze` after ingestion or import.
