import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "pdf" / "arxiv"

QUERIES_FILE = DATA_DIR / "queries.json"
ANSWERS_FILE = DATA_DIR / "answers.json"
QRELS_FILE = DATA_DIR / "qrels.json"

OUTPUT_FILE = BASE_DIR / "generation_eval_150.csv"

SAMPLE_SIZE = 150
RANDOM_SEED = 42


def main():

    print("=" * 70)
    print("CREATING GENERATION EVALUATION SUBSET")
    print("=" * 70)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
        answers = json.load(f)

    with open(QRELS_FILE, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    rows = []

    for query_id, query_data in queries.items():

        if query_id not in answers:
            continue

        if query_id not in qrels:
            continue

        rows.append({
            "query_id": query_id,
            "question": query_data["query"],
            "question_type": query_data.get("type"),
            "source": query_data.get("source"),
            "gold_answer": answers[query_id],
            "gold_doc_id": qrels[query_id]["doc_id"],
            "gold_section_id": qrels[query_id]["section_id"],
        })

    df = pd.DataFrame(rows)

    print(f"\nEligible questions: {len(df)}")

    # --------------------------------------------------------
    # Stratify by question type × source
    # --------------------------------------------------------

    groups = (
        df.groupby(
            ["question_type", "source"],
            group_keys=False
        )
    )

    # Proportional allocation
    allocation = []

    for (question_type, source), group in groups:

        proportion = len(group) / len(df)

        n = round(
            proportion * SAMPLE_SIZE
        )

        allocation.append(
            (
                question_type,
                source,
                n
            )
        )

    # Correct rounding so total = SAMPLE_SIZE
    current_total = sum(
        n for _, _, n in allocation
    )

    difference = SAMPLE_SIZE - current_total

    if difference != 0:

        # Add/subtract from largest group
        largest_index = max(
            range(len(allocation)),
            key=lambda i: allocation[i][2]
        )

        qtype, source, n = allocation[
            largest_index
        ]

        allocation[largest_index] = (
            qtype,
            source,
            n + difference
        )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    selected = []

    for question_type, source, n in allocation:

        group = df[
            (df["question_type"] == question_type)
            &
            (df["source"] == source)
        ]

        sampled = group.sample(
            n=min(n, len(group)),
            random_state=RANDOM_SEED
        )

        selected.append(sampled)

    subset = pd.concat(
        selected,
        ignore_index=True
    )

    # Shuffle final dataset
    subset = subset.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    subset.insert(
        0,
        "evaluation_id",
        [
            f"G{i:03d}"
            for i in range(1, len(subset) + 1)
        ]
    )

    subset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nSelected questions:", len(subset))

    print("\nQuestion type:")
    print(
        subset["question_type"]
        .value_counts()
        .to_string()
    )

    print("\nSource:")
    print(
        subset["source"]
        .value_counts()
        .to_string()
    )

    print("\nType × Source:")
    print(
        pd.crosstab(
            subset["question_type"],
            subset["source"]
        )
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()