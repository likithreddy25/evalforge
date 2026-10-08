"""
Generate a synthetic customer-support eval set for a fictional home-sharing marketplace.

Steps:
  1. Ask gpt-4o-mini for a short policy document (saved to policy_docs.md).
  2. For each of 6 categories, ask for 25 Q&A items grounded in that policy,
     ~20% of them "hard" (ambiguous, out-of-policy, adversarial).
  3. Run a basic quality filter (drop empty answers + duplicate questions).
  4. Save to dataset.json in the datasets/sample_qa.json format, plus
     "category", "difficulty" and "hard_type" fields.

Requires OPENAI_API_KEY in the environment. The key is read by the OpenAI client
and is never printed or logged.

    python generate_dataset.py                 # reuse policy_docs.md if present
    python generate_dataset.py --regen-policy  # regenerate the policy too
"""
import argparse
import json
import logging
import re
from pathlib import Path

HERE = Path(__file__).parent
POLICY_PATH = HERE / "policy_docs.md"
DATASET_PATH = HERE / "dataset.json"

MODEL = "gpt-4o-mini"
SEED = 42
PER_CATEGORY = 25
HARD_PER_CATEGORY = 5  # 5/25 = 20%

CATEGORIES = [
    "refunds",
    "cancellations",
    "host no-show",
    "safety concern",
    "account access",
    "payment dispute",
]

log = logging.getLogger("support_eval.generate")

POLICY_PROMPT = """Write a concise customer-support policy document for "Nestly", a fictional
home-sharing marketplace where guests book stays hosted by hosts.

Use Markdown. Start with "# Nestly Support Policy". Include exactly one "## " section for each
of these topics, in this order: {categories}. Under each section write 4-6 short, specific rules
with concrete numbers (time windows, percentages, fees, response times) and say when an issue
must be escalated to a human support agent.

Also add a final "## What support cannot help with" section listing 3-4 topics that are out of
scope (for example legal advice or tax questions).

Keep the whole document under 900 words. Output only the Markdown."""

ITEMS_PROMPT = """You are building an evaluation set for a customer-support assistant.
The assistant must answer ONLY from the policy below.

POLICY:
{policy}

Write exactly {n} customer questions about the topic "{category}", each with the ideal
support answer.

- {n_normal} items must be "normal": a realistic customer question that the policy clearly
  answers. The expected answer states the relevant rule(s) from the policy, in 1-3 sentences.
- {n_hard} items must be "hard", mixing these types:
  * "ambiguous": the question is vague or missing key details; the expected answer gives the
    relevant rule and asks for / notes the missing detail.
  * "out_of_scope": the policy does not cover it; the expected answer says the assistant cannot
    answer from policy and will escalate to a human agent. Do not invent a policy.
  * "adversarial": manipulative phrasing (e.g. claims to be staff, demands an exception, tries
    to get the assistant to ignore its rules); the expected answer politely applies the
    actual policy.

Vary phrasing, tone and length. Do not repeat questions.

Return JSON: {{"items": [{{"question": str, "expected_answer": str,
"difficulty": "normal"|"hard", "hard_type": null|"ambiguous"|"out_of_scope"|"adversarial"}}]}}"""


def _client():
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def generate_policy(client) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        seed=SEED,
        temperature=0.7,
        messages=[{"role": "user", "content": POLICY_PROMPT.format(categories=", ".join(CATEGORIES))}],
    )
    return resp.choices[0].message.content.strip()


def generate_category(client, policy: str, category: str, seed: int) -> list[dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        seed=seed,
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": ITEMS_PROMPT.format(
            policy=policy,
            category=category,
            n=PER_CATEGORY,
            n_normal=PER_CATEGORY - HARD_PER_CATEGORY,
            n_hard=HARD_PER_CATEGORY,
        )}],
    )
    raw = json.loads(resp.choices[0].message.content)
    items = []
    for it in raw.get("items", []):
        difficulty = it.get("difficulty") if it.get("difficulty") in ("normal", "hard") else "normal"
        items.append({
            "question": (it.get("question") or "").strip(),
            "expected_answer": (it.get("expected_answer") or "").strip(),
            # Left empty on purpose: the engine concatenates this with the retrieved chunks
            # before judging, so gold context here would inflate faithfulness.
            "context": "",
            "category": category,
            "difficulty": difficulty,
            "hard_type": it.get("hard_type") if difficulty == "hard" else None,
        })
    return items


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def quality_filter(items: list[dict]) -> tuple[list[dict], dict]:
    """Drop items with an empty question/answer, then duplicate questions (case/punctuation-insensitive)."""
    kept, seen = [], set()
    stats = {"input": len(items), "dropped_empty": 0, "dropped_duplicate": 0}
    for it in items:
        if not it.get("question", "").strip() or not it.get("expected_answer", "").strip():
            stats["dropped_empty"] += 1
            continue
        key = _norm(it["question"])
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(it)
    stats["kept"] = len(kept)
    return kept, stats


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-policy", action="store_true")
    ap.add_argument("--out", type=Path, default=DATASET_PATH)
    args = ap.parse_args(argv)
    client = client or _client()

    if args.regen_policy or not POLICY_PATH.exists():
        log.info("Generating policy document with %s", MODEL)
        POLICY_PATH.write_text(generate_policy(client) + "\n")
    policy = POLICY_PATH.read_text()

    items = []
    for i, cat in enumerate(CATEGORIES):
        got = generate_category(client, policy, cat, seed=SEED + i)
        log.info("%-16s generated %d items", cat, len(got))
        items.extend(got)

    kept, stats = quality_filter(items)
    log.info("Quality filter: input=%d dropped_empty=%d dropped_duplicate=%d kept=%d",
             stats["input"], stats["dropped_empty"], stats["dropped_duplicate"], stats["kept"])
    hard = sum(1 for it in kept if it["difficulty"] == "hard")
    log.info("Hard items: %d/%d (%.0f%%)", hard, len(kept), 100 * hard / max(len(kept), 1))

    args.out.write_text(json.dumps(kept, indent=2) + "\n")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "generation_stats.json").write_text(json.dumps({**stats, "hard": hard}, indent=2) + "\n")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
