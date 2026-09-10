import pandas as pd
import re

INPUT = "results_b0.csv"
BENCHMARK = "questions_validated.csv"
OUTPUT = "b0_scored.csv"


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\*\*", "", text)

    # Fix common encoding issues
    text = text.replace("â€“", "-")
    text = text.replace("â€”", "-")
    text = text.replace("â€™", "'")

    return " ".join(text.split())


def extract_numbers(text):
    text = normalize(text)

    return re.findall(
        r"\d+(?:\.\d+)?",
        text
    )


def is_idk(text):
    if pd.isna(text):
        return False

    text = normalize(text)

    phrases = [
        "i don't know",
        "i do not know",
        "i’m sorry",
        "i'm sorry",
    ]

    return any(p in text for p in phrases)


def is_error(status, answer):
    status = normalize(status)
    answer = normalize(answer)

    return (
        status == "error"
        or answer.startswith("error:")
        or "rate limit reached" in answer
        or "rate_limit_exceeded" in answer
    )


def numeric_match(expected, answer):

    expected_numbers = extract_numbers(expected)
    answer_numbers = extract_numbers(answer)

    if not expected_numbers:
        return False

    return all(
        number in answer_numbers
        for number in expected_numbers
    )


def keyword_match(expected, answer):

    expected = normalize(expected)
    answer = normalize(answer)

    words = re.findall(
        r"[a-zA-Z][a-zA-Z-]+",
        expected
    )

    stopwords = {
        "the", "and", "of", "to", "a", "an",
        "in", "for", "was", "were", "is",
        "are", "with", "on", "by", "from",
        "per", "what", "which", "how",
        "that", "this", "their", "they",
        "according"
    }

    keywords = [
        word
        for word in words
        if word not in stopwords and len(word) > 2
    ]

    if not keywords:
        return False

    matched = sum(
        word in answer
        for word in keywords
    )

    return matched / len(keywords) >= 0.60


def main():

    results = pd.read_csv(INPUT)
    benchmark = pd.read_csv(BENCHMARK)

    result_columns = [
        "question_id",
        "rag_answer_b0",
        "response_time_seconds",
        "api_status",
    ]

    df = benchmark.merge(
        results[result_columns],
        on="question_id",
        how="left"
    )

    # ---------------------------------------------------------
    # STATUS CLASSIFICATION
    # ---------------------------------------------------------

    df["evaluation_status"] = "OK"

    df.loc[
        df["api_status"].fillna("").str.upper() != "OK",
        "evaluation_status"
    ] = "ERROR"

    df.loc[
        df.apply(
            lambda row: is_error(
                row["api_status"],
                row["rag_answer_b0"]
            ),
            axis=1
        ),
        "evaluation_status"
    ] = "ERROR"

    df.loc[
        (
            df["evaluation_status"] == "OK"
        )
        &
        (
            df["rag_answer_b0"].isna()
            |
            (df["rag_answer_b0"].astype(str).str.strip() == "")
        ),
        "evaluation_status"
    ] = "MISSING"

    # ---------------------------------------------------------
    # IDK
    # ---------------------------------------------------------

    df["idk"] = df["rag_answer_b0"].apply(is_idk)

    df.loc[
        df["evaluation_status"] != "OK",
        "idk"
    ] = False

    # ---------------------------------------------------------
    # MATCHING
    # ---------------------------------------------------------

    df["numeric_match"] = [
        numeric_match(expected, answer)
        if status == "OK"
        else False
        for expected, answer, status
        in zip(
            df["expected_answer"],
            df["rag_answer_b0"],
            df["evaluation_status"]
        )
    ]

    df["keyword_match"] = [
        keyword_match(expected, answer)
        if status == "OK"
        else False
        for expected, answer, status
        in zip(
            df["expected_answer"],
            df["rag_answer_b0"],
            df["evaluation_status"]
        )
    ]

    # ---------------------------------------------------------
    # AUTOMATIC CORRECTNESS
    # ---------------------------------------------------------

    is_numerical = df["question_type"].str.contains(
        "Numerical",
        case=False,
        na=False
    )

    df["automatic_correct"] = (
        (df["evaluation_status"] == "OK")
        &
        (~df["idk"])
        &
        (
            (is_numerical & df["numeric_match"])
            |
            (~is_numerical & df["keyword_match"])
        )
    )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    total = len(df)

    valid = df["evaluation_status"] == "OK"

    valid_count = int(valid.sum())

    correct = int(df["automatic_correct"].sum())

    idk_count = int(
        (
            valid
            &
            df["idk"]
        ).sum()
    )

    errors = int(
        (df["evaluation_status"] == "ERROR").sum()
    )

    missing = int(
        (df["evaluation_status"] == "MISSING").sum()
    )

    accuracy_all = (
        correct / total * 100
        if total
        else 0
    )

    accuracy_valid = (
        correct / valid_count * 100
        if valid_count
        else 0
    )

    idk_rate_valid = (
        idk_count / valid_count * 100
        if valid_count
        else 0
    )

    avg_time = (
        df.loc[valid, "response_time_seconds"].mean()
        if valid_count
        else 0
    )

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print("=" * 70)
    print("B0 VALIDATED BASELINE")
    print("=" * 70)

    print(f"Total questions        : {total}")
    print(f"Valid responses        : {valid_count}")
    print(f"Correct                : {correct}")

    print(
        f"Accuracy (all)         : "
        f"{accuracy_all:.2f}%"
    )

    print(
        f"Accuracy (valid only)  : "
        f"{accuracy_valid:.2f}%"
    )

    print(f"I don't know           : {idk_count}")

    print(
        f"IDK rate (valid only)  : "
        f"{idk_rate_valid:.2f}%"
    )

    print(f"Errors / rate limits   : {errors}")
    print(f"Missing responses      : {missing}")

    print(
        f"Average response time  : "
        f"{avg_time:.2f} seconds"
    )

    print()
    print("=" * 70)
    print("STATUS BREAKDOWN")
    print("=" * 70)

    print(
        df["evaluation_status"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 70)
    print("RESULTS BY QUESTION")
    print("=" * 70)

    print(
        df[
            [
                "question_id",
                "evaluation_status",
                "automatic_correct",
                "idk",
                "numeric_match",
                "keyword_match",
            ]
        ].to_string(index=False)
    )

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()