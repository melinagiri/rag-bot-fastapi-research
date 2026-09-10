import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

QUESTIONS_FILE = BASE_DIR / "generation_eval_60.csv"
EMBEDDINGS_FILE = BASE_DIR / "corpus_embeddings.npy"
METADATA_FILE = BASE_DIR / "corpus_metadata.json"

OUTPUT_DIR = BASE_DIR / "generation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_DIM = 384
OPTIMIZED_CONTEXT_SIZE = 5
MAX_CONTEXT_CHARACTERS = 8000
MAX_OUTPUT_TOKENS = 512

ENV_FILE = PROJECT_DIR / ".env"
load_dotenv(ENV_FILE)

print("=" * 70)
print("FINAL RAG GENERATION EXPERIMENT")
print("=" * 70)
print(f"\nModel: {MODEL_NAME}")


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=MAX_OUTPUT_TOKENS,
)

print("\nLoading generation subset...")
questions = pd.read_csv(QUESTIONS_FILE)
print(f"Questions loaded: {len(questions)}")


print("\nLoading corpus embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)
print(f"Embeddings shape: {embeddings.shape}")


print("\nLoading corpus metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Metadata records: {len(metadata)}")


if len(embeddings) != len(metadata):
    raise RuntimeError(
        "Embedding count and metadata count do not match."
    )

if embeddings.shape[1] != EMBEDDING_DIM:
    raise RuntimeError(
        f"Expected embedding dimension {EMBEDDING_DIM}, "
        f"got {embeddings.shape[1]}"
    )


print("\nNormalizing embeddings...")

embedding_norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

embedding_norms[embedding_norms == 0] = 1
normalized_embeddings = embeddings / embedding_norms

print("\nLoading query embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-v2"
)


def retrieve(question, k):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    scores = normalized_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        item = metadata[int(index)]

        text = item.get("text", "")

        results.append({
            "rank": rank,
            "index": int(index),
            "score": float(scores[index]),
            "doc_id": item.get("doc_id"),
            "section_id": item.get("section_id"),
            "text": text
        })

    return results


def optimize_context(retrieved, max_sections=5):

    selected = []

    for item in retrieved:

        text = item["text"].strip()

        if not text:
            continue

        words = set(text.lower().split())

        duplicate = False

        for selected_item in selected:

            selected_words = set(
                selected_item["text"].lower().split()
            )

            if not words or not selected_words:
                continue

            overlap = (
                len(words & selected_words)
                / len(words | selected_words)
            )

            if overlap >= 0.70:
                duplicate = True
                break

        if not duplicate:
            selected.append(item)

        if len(selected) >= max_sections:
            break

    return selected

def limit_context_characters(context_items, max_characters=MAX_CONTEXT_CHARACTERS):

    selected = []
    total = 0

    for item in context_items:

        text = item["text"].strip()

        if not text:
            continue

        remaining = max_characters - total

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining]

        new_item = item.copy()
        new_item["text"] = text

        selected.append(new_item)

        total += len(text)

    return selected

SYSTEM_PROMPT = """
You are answering questions using retrieved evidence.

Rules:
1. Answer using the supplied context.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information, say:
"I don't know."
4. Give a concise but complete answer.
"""


def build_prompt(question, context_items):

    context_parts = []

    for item in context_items:

        context_parts.append(
            f"[Evidence {item['rank']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    return (
        SYSTEM_PROMPT
        + "\n\nCONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

def generate_answer(question, context_items):

    prompt = build_prompt(
        question,
        context_items
    )

    start = time.perf_counter()

    response = llm.invoke(prompt)

    elapsed = time.perf_counter() - start

    answer = response.content

    if isinstance(answer, list):
        answer = " ".join(
            str(x) for x in answer
        )

    return str(answer), elapsed

def run_condition(condition_name, k, optimize=False):

    output_file = OUTPUT_DIR / f"{condition_name}.csv"

    print("\n")
    print("=" * 70)
    print(f"CONDITION: {condition_name}")
    print("=" * 70)

    completed = {}

    if output_file.exists():

        old = pd.read_csv(output_file)

        for _, row in old.iterrows():
            completed[str(row["evaluation_id"])] = row.to_dict()

        print(f"Existing results found: {len(completed)}")

    results = list(completed.values())

    for position, row in questions.iterrows():

        evaluation_id = str(row["evaluation_id"])

        if evaluation_id in completed:

            print(
                f"[{position + 1}/{len(questions)}] "
                f"{evaluation_id} - SKIP"
            )

            continue

        question = str(row["question"])

        print(
            f"\n[{position + 1}/{len(questions)}] "
            f"{evaluation_id}"
        )

        print(question)

        try:

            retrieved = retrieve(question, k)

            if optimize:
                context = optimize_context(
                    retrieved,
                    OPTIMIZED_CONTEXT_SIZE
                )
            else:
                context = retrieved

            context = limit_context_characters(
                context,
                MAX_CONTEXT_CHARACTERS
            )

            context_characters = sum(
                len(x["text"])
                for x in context
            )

            print(
                f"Retrieved: {len(retrieved)} | "
                f"Context: {len(context)} sections | "
                f"Characters: {context_characters}"
            )

            if context_characters == 0:

                raise RuntimeError(
                    "Retrieved metadata contains no text. "
                    "Check corpus_metadata.json."
                )

            answer, generation_time = generate_answer(
                question,
                context
            )

            result = {

                "evaluation_id":
                    evaluation_id,

                "question":
                    question,

                "question_type":
                    row["question_type"],

                "source":
                    row["source"],

                "gold_answer":
                    row["gold_answer"],

                "condition":
                    condition_name,

                "retrieval_k":
                    k,

                "optimization":
                    optimize,

                "context_sections":
                    len(context),

                "context_characters":
                    context_characters,

                "retrieved_doc_ids":
                    json.dumps(
                        [
                            x["doc_id"]
                            for x in retrieved
                        ]
                    ),

                "context_doc_ids":
                    json.dumps(
                        [
                            x["doc_id"]
                            for x in context
                        ]
                    ),

                "answer":
                    answer,

                "generation_time_seconds":
                    round(generation_time, 3),

                "status":
                    "OK"
            }

        except Exception as e:

            print(f"ERROR: {e}")

            result = {

                "evaluation_id":
                    evaluation_id,

                "question":
                    question,

                "question_type":
                    row["question_type"],

                "source":
                    row["source"],

                "gold_answer":
                    row["gold_answer"],

                "condition":
                    condition_name,

                "retrieval_k":
                    k,

                "optimization":
                    optimize,

                "context_sections":
                    0,

                "context_characters":
                    0,

                "retrieved_doc_ids":
                    "[]",

                "context_doc_ids":
                    "[]",

                "answer":
                    "",

                "generation_time_seconds":
                    0,

                "status":
                    f"ERROR: {e}"
            }

        results.append(result)

        pd.DataFrame(results).to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Saved: {output_file.name}")

    print(
        f"\nCompleted condition: {condition_name}"
    )

    print(f"Results: {output_file}")

def main():

    print("\nStarting experiments.")

    run_condition(
        condition_name="R0_k3",
        k=3,
        optimize=False
    )

    run_condition(
        condition_name="R1_k10",
        k=10,
        optimize=False
    )

    run_condition(
        condition_name="R2_k10_optimized",
        k=10,
        optimize=True
    )

    print("\n")
    print("=" * 70)
    print("ALL GENERATION EXPERIMENTS COMPLETE")
    print("=" * 70)

    print(
        f"\nResults directory:\n{OUTPUT_DIR}"
    )

if __name__ == "__main__":
    main()
