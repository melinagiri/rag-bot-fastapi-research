import os
import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

QUESTIONS_FILE = "questions.csv"
# OUTPUT_FILE = "results_b0.csv"
OUTPUT_FILE = "results_b1.csv"

MODEL_PROVIDER = "Groq"
MODEL_NAME = "openai/gpt-oss-120b"


def ask_rag(message):

    payload = {
        "model_provider": MODEL_PROVIDER,
        "model_name": MODEL_NAME,
        "message": message
    }

    response = requests.post(
        f"{API_URL}/chat",
        json=payload,
        timeout=180
    )

    if not response.ok:
        print("API ERROR:")
        print(response.text)
        response.raise_for_status()

    data = response.json()

    # The API returns StandardAPIResponse:
    # {status, data, message}

    if isinstance(data, dict):

        if data.get("status") == "error":
            raise RuntimeError(data.get("message"))

        # Most likely answer is inside data["data"]
        answer_data = data.get("data")

        if isinstance(answer_data, str):
            return answer_data

        if isinstance(answer_data, dict):
            for key in ["answer", "response", "message", "content"]:
                if key in answer_data:
                    return answer_data[key]

        if answer_data is not None:
            return str(answer_data)

        if data.get("message"):
            return data["message"]

    return str(data)


def main():

    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(
            f"{QUESTIONS_FILE} was not found."
        )

    df = pd.read_csv(QUESTIONS_FILE)

    print(f"Loaded {len(df)} benchmark questions.")
    print("Columns:")
    print(list(df.columns))
    print()

    results = []

    for index, row in df.iterrows():

        question_id = str(row["question_id"])
        question = str(row["question"])

        print("=" * 70)
        print(f"[{index + 1}/{len(df)}] {question_id}")
        print(question)

        start = time.perf_counter()

        try:
            answer = ask_rag(question)
            status = "OK"

        except Exception as e:
            answer = f"ERROR: {e}"
            status = "ERROR"

        elapsed = time.perf_counter() - start

        result = row.to_dict()

        result["rag_answer_b0"] = answer
        result["response_time_seconds"] = round(elapsed, 3)
        result["api_status"] = status

        results.append(result)

        # Save continuously
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Status: {status}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Answer: {answer}")
        print()

    print("=" * 70)
    print("BASELINE B0 COMPLETE")
    print("=" * 70)
    print(f"Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()