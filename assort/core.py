import json
from typing import List, Dict
from openai import OpenAI

_MODEL = "gpt-4o-mini"
_client = OpenAI()


def _suggest_categories(
    conversations: List[str], min_clusters: int, max_clusters: int
) -> List[str]:
    prompt = (
        "You are a conversation categorizer. "
        f"Return between {min_clusters} and {max_clusters} category names that best group these texts "
        "as a JSON array of strings."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n".join(conversations)},
    ]
    response = _client.chat.completions.create(
        model=_MODEL, messages=messages, temperature=0
    )
    return json.loads(response.choices[0].message.content)


def _assign(conversation: str, categories: List[str]) -> str:
    prompt = (
        "Choose the single best category for the given text from this list: "
        f"{', '.join(categories)}. "
        'Respond as {"category": "<name>"}.'
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": conversation},
    ]
    response = _client.chat.completions.create(
        model=_MODEL, messages=messages, temperature=0
    )
    return json.loads(response.choices[0].message.content)["category"]


def categorize(
    conversations: List[str], min_clusters: int = 2, max_clusters: int = 5
) -> Dict[str, List[int]]:
    categories = _suggest_categories(conversations, min_clusters, max_clusters)
    clusters = {c: [] for c in categories}
    for idx, text in enumerate(conversations):
        cat = _assign(text, categories)
        clusters.setdefault(cat, []).append(idx)
    return clusters
