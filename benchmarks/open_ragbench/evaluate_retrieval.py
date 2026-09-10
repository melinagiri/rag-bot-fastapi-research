import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "pdf" / "arxiv"
CORPUS_DIR = DATA_DIR / "corpus"

QUERIES_FILE = DATA_DIR / "queries.json"
QRELS_FILE = DATA_DIR / "qrels.json"

EMBEDDINGS_FILE = BASE_DIR / "corpus_embeddings.npy"
METADATA_FILE = BASE_DIR / "corpus_metadata.json"


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Open RAG Benchmark retrieval evaluation"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of retrieved sections"
    )

    parser.add_argument(
        "--run",
        type=str,
        default="b0",
        help="Experiment/run name"
    )

    return parser.parse_args()


# ============================================================
# JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CORPUS
# ============================================================

def load_corpus():

    records = []

    files = sorted(CORPUS_DIR.glob("*.json"))

    print(f"Corpus files found: {len(files)}")

    for file_path in files:

        doc_id = file_path.stem

        with open(file_path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        sections = paper.get("sections", [])

        for section_id, section in enumerate(sections):

            text = section.get("text", "")

            if not text:
                continue

            records.append({
                "doc_id": doc_id,
                "section_id": section_id,
                "text": text,
            })

    return records


# ============================================================
# CORPUS EMBEDDINGS
# ============================================================

def build_or_load_embeddings(corpus, model):

    # --------------------------------------------------------
    # Load cached embeddings if available
    # --------------------------------------------------------

    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():

        print("\nCached embeddings found.")

        embeddings = np.load(
            EMBEDDINGS_FILE
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        print(
            f"Loaded cached embeddings: "
            f"{embeddings.shape}"
        )

        if len(metadata) != len(corpus):

            print(
                "WARNING: Cache size does not match "
                "current corpus."
            )

            print("Rebuilding embeddings...")

        else:

            return embeddings

    # --------------------------------------------------------
    # Build embeddings
    # --------------------------------------------------------

    print("\nNo valid cache found.")
    print("Embedding corpus sections...")

    start = time.perf_counter()

    corpus_texts = [
        record["text"]
        for record in corpus
    ]

    embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    elapsed = time.perf_counter() - start

    print(
        f"Corpus embedding time: "
        f"{elapsed:.2f} seconds"
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    metadata = [
        {
            "doc_id": record["doc_id"],
            "section_id": record["section_id"]
        }
        for record in corpus
    ]

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved embeddings to: "
        f"{EMBEDDINGS_FILE}"
    )

    print(
        f"Saved metadata to: "
        f"{METADATA_FILE}"
    )

    return embeddings


# ============================================================
# METRICS
# ============================================================

def reciprocal_rank(ranked_sections, gold):

    for rank, item in enumerate(
        ranked_sections,
        start=1
    ):

        if item == gold:
            return 1.0 / rank

    return 0.0


def hit_at_k(ranked_sections, gold, k):

    return int(
        gold in ranked_sections[:k]
    )


def ndcg_at_k(ranked_sections, gold, k):

    for rank, item in enumerate(
        ranked_sections[:k],
        start=1
    ):

        if item == gold:

            return 1.0 / np.log2(
                rank + 1
            )

    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    k = args.k
    run_name = args.run

    print("=" * 70)
    print("OPEN RAG BENCHMARK")
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nRun name       : {run_name}")
    print(f"Retrieval K    : {k}")
    print(f"Embedding      : {EMBEDDING_MODEL}")

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    print("\nLoading queries...")

    queries = load_json(
        QUERIES_FILE
    )

    print(
        f"Queries loaded: {len(queries)}"
    )

    print("\nLoading qrels...")

    qrels = load_json(
        QRELS_FILE
    )

    print(
        f"Qrels loaded: {len(qrels)}"
    )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    start = time.perf_counter()

    corpus = load_corpus()

    elapsed = time.perf_counter() - start

    print(
        f"Corpus sections: {len(corpus)}"
    )

    print(
        f"Corpus loading time: "
        f"{elapsed:.2f} seconds"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    corpus_embeddings = (
        build_or_load_embeddings(
            corpus,
            model
        )
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nStarting retrieval evaluation...")

    results = []

    start = time.perf_counter()

    for index, (
        query_id,
        query_data
    ) in enumerate(
        queries.items(),
        start=1
    ):

        question = query_data["query"]

        gold_info = qrels.get(
            query_id
        )

        if not gold_info:

            print(
                f"WARNING: Missing qrel "
                f"for {query_id}"
            )

            continue

        gold = (
            gold_info["doc_id"],
            gold_info["section_id"]
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Exact cosine similarity
        # ----------------------------------------------------

        scores = np.dot(
            corpus_embeddings,
            query_embedding
        )

        # ----------------------------------------------------
        # Get enough results for evaluation
        # ----------------------------------------------------

        ranking = np.argsort(
            scores
        )[::-1]

        ranked_sections = [
            (
                corpus[i]["doc_id"],
                corpus[i]["section_id"]
            )
            for i in ranking[:k]
        ]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        rr = reciprocal_rank(
            ranked_sections,
            gold
        )

        row = {
            "run": run_name,
            "k": k,
            "query_id": query_id,
            "question": question,
            "query_type": query_data.get("type"),
            "query_source": query_data.get("source"),
            "gold_doc_id": gold_info["doc_id"],
            "gold_section_id": gold_info["section_id"],
            "reciprocal_rank": rr,
            "hit_at_k": hit_at_k(
                ranked_sections,
                gold,
                k
            ),
            "ndcg_at_k": ndcg_at_k(
                ranked_sections,
                gold,
                k
            ),
            "top_results": str(
                ranked_sections
            ),
        }

        results.append(row)

        if (
            index % 100 == 0
            or index == len(queries)
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"Processed "
                f"{index}/{len(queries)} "
                f"queries | "
                f"{elapsed:.1f}s"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    output_file = (
        BASE_DIR
        / f"retrieval_{run_name}_k{k}.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(f"RESULTS: {run_name}")
    print("=" * 70)

    print(
        f"Questions evaluated : "
        f"{len(df)}"
    )

    print(
        f"Hit@{k}              : "
        f"{df['hit_at_k'].mean():.4f}"
    )

    print(
        f"MRR                 : "
        f"{df['reciprocal_rank'].mean():.4f}"
    )

    print(
        f"nDCG@{k}             : "
        f"{df['ndcg_at_k'].mean():.4f}"
    )

    print(
        f"\nSaved:"
        f"\n{output_file}"
    )


if __name__ == "__main__":
    main()