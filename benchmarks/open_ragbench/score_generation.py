import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "generation_results"

FILES = {
    "R0_k3": RESULTS_DIR / "R0_k3.csv",
    "R1_k10": RESULTS_DIR / "R1_k10.csv",
    "R2_k10_optimized": RESULTS_DIR / "R2_k10_optimized.csv",
}

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text

def token_set(text):
    text = normalize(text)

    return set(
        re.findall(r"[a-zA-Z0-9]+", text)
    )

def lexical_f1(gold, answer):

    gold_tokens = token_set(gold)
    answer_tokens = token_set(answer)

    if not gold_tokens or not answer_tokens:
        return 0.0

    overlap = len(
        gold_tokens & answer_tokens
    )

    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )

def is_idk(answer):

    answer = normalize(answer)

    phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "cannot determine",
        "not enough information"
    ]

    return any(
        phrase in answer
        for phrase in phrases
    )

def load_result(name, path):

    print(f"\nLoading {name}")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "evaluation_id",
        "question",
        "question_type",
        "source",
        "gold_answer",
        "answer",
        "generation_time_seconds",
        "context_sections",
        "context_characters",
        "status"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing columns: "
            f"{missing}"
        )

    df["gold_answer"] = (
        df["gold_answer"]
        .fillna("")
    )

    df["answer"] = (
        df["answer"]
        .fillna("")
    )

    df["lexical_f1"] = [
        lexical_f1(
            gold,
            answer
        )
        for gold, answer
        in zip(
            df["gold_answer"],
            df["answer"]
        )
    ]

    df["idk"] = df["answer"].apply(
        is_idk
    )

    return df

def calculate_summary(df, name):

    successful = df[
        df["status"]
        .astype(str)
        .str.startswith("OK")
    ].copy()

    if len(successful) == 0:

        return {
            "condition": name,
            "questions": len(df),
            "successful": 0,
            "errors": len(df),
            "mean_lexical_f1": 0,
            "idk_percent": 0,
            "mean_latency_seconds": 0,
            "mean_context_sections": 0,
            "mean_context_characters": 0
        }

    return {

        "condition":
            name,

        "questions":
            len(df),

        "successful":
            len(successful),

        "errors":
            len(df) - len(successful),

        "mean_lexical_f1":
            round(
                successful[
                    "lexical_f1"
                ].mean(),
                4
            ),

        "idk_percent":
            round(
                successful[
                    "idk"
                ].mean() * 100,
                2
            ),

        "mean_latency_seconds":
            round(
                successful[
                    "generation_time_seconds"
                ].mean(),
                3
            ),

        "mean_context_sections":
            round(
                successful[
                    "context_sections"
                ].mean(),
                2
            ),

        "mean_context_characters":
            round(
                successful[
                    "context_characters"
                ].mean(),
                0
            )
    }

def subgroup_summary(
    df,
    name,
    column
):

    rows = []

    successful = df[
        df["status"]
        .astype(str)
        .str.startswith("OK")
    ].copy()

    for value, group in successful.groupby(
        column
    ):

        rows.append({

            "condition":
                name,

            column:
                value,

            "questions":
                len(group),

            "mean_lexical_f1":
                round(
                    group[
                        "lexical_f1"
                    ].mean(),
                    4
                ),

            "idk_percent":
                round(
                    group[
                        "idk"
                    ].mean() * 100,
                    2
                ),

            "mean_latency_seconds":
                round(
                    group[
                        "generation_time_seconds"
                    ].mean(),
                    3
                )
        })

    return rows

def main():

    print("=" * 70)
    print("GENERATION EXPERIMENT SCORING")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    datasets = {}

    for name, path in FILES.items():

        datasets[name] = load_result(
            name,
            path
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_rows = []

    for name, df in datasets.items():

        summary_rows.append(
            calculate_summary(
                df,
                name
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Question type
    # --------------------------------------------------------

    type_rows = []

    for name, df in datasets.items():

        type_rows.extend(
            subgroup_summary(
                df,
                name,
                "question_type"
            )
        )

    by_type = pd.DataFrame(
        type_rows
    )

    print("\n")
    print("=" * 70)
    print("RESULTS BY QUESTION TYPE")
    print("=" * 70)

    print(
        by_type.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source_rows = []

    for name, df in datasets.items():

        source_rows.extend(
            subgroup_summary(
                df,
                name,
                "source"
            )
        )

    by_source = pd.DataFrame(
        source_rows
    )

    print("\n")
    print("=" * 70)
    print("RESULTS BY SOURCE")
    print("=" * 70)

    print(
        by_source.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Combined results
    # --------------------------------------------------------

    combined = pd.concat(
        [
            df.assign(
                condition=name
            )
            for name, df
            in datasets.items()
        ],
        ignore_index=True
    )

    combined_file = (
        RESULTS_DIR
        / "generation_comparison.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------

    summary_file = (
        RESULTS_DIR
        / "generation_summary.csv"
    )

    type_file = (
        RESULTS_DIR
        / "generation_by_question_type.csv"
    )

    source_file = (
        RESULTS_DIR
        / "generation_by_source.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    by_type.to_csv(
        type_file,
        index=False,
        encoding="utf-8-sig"
    )

    by_source.to_csv(
        source_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Paired comparison
    # --------------------------------------------------------

    pivot = combined.pivot(
        index="evaluation_id",
        columns="condition",
        values="lexical_f1"
    )

    print("\n")
    print("=" * 70)
    print("PAIRED LEXICAL-F1 DIFFERENCES")
    print("=" * 70)

    comparisons = [
        ("R0_k3", "R1_k10"),
        (
            "R1_k10",
            "R2_k10_optimized"
        ),
        (
            "R0_k3",
            "R2_k10_optimized"
        )
    ]

    for first, second in comparisons:

        if (
            first in pivot.columns
            and second in pivot.columns
        ):

            difference = (
                pivot[second]
                - pivot[first]
            ).dropna()

            print(
                f"{first} -> {second}: "
                f"{difference.mean():+.4f}"
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SCORING COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved:\n"
        f"{summary_file}\n"
        f"{type_file}\n"
        f"{source_file}\n"
        f"{combined_file}"
    )

    print("\nNOTE:")
    print(
        "Lexical F1 is an automatic screening metric. "
        "It should not be presented as definitive semantic "
        "answer correctness."
    )

if __name__ == "__main__":
    main()
