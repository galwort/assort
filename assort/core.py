from enum import Enum
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, create_model
from random import choice
from requests import get
from tiktoken import encoding_for_model
from time import sleep
from typing import List, Dict
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class CategoryModel(BaseModel):
    categories: List[str]


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Policy(str, Enum):
    fuzzy = "fuzzy"
    exhaustive = "exhaustive"


_MODEL = "gpt-4o-mini"
_client = OpenAI()


def _get_model_info(_MODEL: str) -> Dict[str, str]:
    model_json = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    try:
        response = get(model_json)
        response.raise_for_status()
        model_info = response.json()
        context_window = model_info.get(_MODEL, {}).get("context_window", 0)
        cost_per_input_token = model_info.get(_MODEL, {}).get(
            "cost_per_input_token", 0.0
        )
        cost_per_output_token = model_info.get(_MODEL, {}).get(
            "cost_per_output_token", 0.0
        )
        return {
            "context_window": context_window,
            "cost_per_input_token": cost_per_input_token,
            "cost_per_output_token": cost_per_output_token,
        }
    except Exception as e:
        print(f"Error fetching model info: {e}")
        return {
            "context_window": 0,
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        }


def _clean_batch(batch: List[str]) -> List[List[str]]:
    model_info = _get_model_info(_MODEL)
    encoder = encoding_for_model(_MODEL)
    context_window = model_info["context_window"]
    max_tokens = context_window / 2

    cleaned_batch = []
    current_batch = []
    token_count = 0

    for text in batch:
        tokens = encoder.encode(text)
        token_len = len(tokens)

        if token_len > max_tokens:
            continue

        if token_count + token_len > max_tokens:
            cleaned_batch.append(current_batch)
            current_batch = []
            token_count = 0

        current_batch.append(text)
        token_count += token_len

    if current_batch:
        cleaned_batch.append(current_batch)

    return cleaned_batch


def _gen_categories(
    batch: List[str], min_clusters: int, max_clusters: int, description: str = ""
) -> List[str]:
    max_clusters = max_clusters - 1
    cleaned_batch = _clean_batch(batch)

    description = (
        "A description of the batch is as follows: " + description
        if description
        else ""
    )

    system_message = (
        "You are a text categorizer. "
        "When given a batch of objects, you are to come up with "
        f"between {min_clusters} and NO MORE THAN {max_clusters} distinct categories "
        "that the objects could be distinctly sorted into. "
        f"{description}\n"
        f"{'-' * 80}\n"
        "Your reply should be in JSON format, with categories as a single key, "
        "followed by a list of categories that the objects would best fit into. "
        f"And remember - do not create more than {max_clusters} categories."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n\n".join(cleaned_batch)},
    ]

    while True:
        try:
            response = _client.responses.parse(
                model=_MODEL, input=messages, text_format=CategoryModel
            )
            return response.output_parsed.categories
        except OpenAIError as e:
            if "rate limit" in str(e).lower():
                sleep(60)
            elif "insufficient_quota" in str(e).lower():
                print(
                    "Account is not funded, check billing at https://platform.openai.com/settings/organization/billing/"
                )
                exit()


def _create_SortModel(category_keys: List[str]) -> BaseModel:
    fields = {key: (ConfidenceLevel, ConfidenceLevel.low) for key in category_keys}
    SortModel = create_model("SortModel", **fields)
    return SortModel


def _gen_sort(text: str, category_keys: List[str], description: str = None) -> str:
    SortModel = _create_SortModel(category_keys)
    description = (
        "The categories are generated from a larger set, which has been described as follows: "
        + description
        if description
        else ""
    )

    system_message = (
        "You are a text sorter. "
        "When given a piece of text, you are to sort it into one of the following categories "
        f"{', '.join(category_keys)}. "
        f"{description}\n"
        "You should also provide a confidence level of high, medium, or low for each category. "
        "Your reply should be in JSON format, with the keys being the category names "
        "and the values being the confidence level for that category."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]

    while True:
        try:
            response = _client.responses.parse(
                model=_MODEL, input=messages, text_format=SortModel
            )
            return response.output_parsed.model_dump()
        except OpenAIError as e:
            if "rate limit" in str(e).lower():
                sleep(60)
            elif "insufficient_quota" in str(e).lower():
                print(
                    "Account is not funded, check billing at https://platform.openai.com/settings/organization/billing/"
                )
                exit()


def assort(
    batch: List[str],
    min_clusters: int = 2,
    max_clusters: int = 5,
    policy: Policy = Policy.fuzzy,
    description: str = "",
) -> Dict[str, List[int]]:
    categories = _gen_categories(batch, min_clusters, max_clusters, description)
    sorted_results = {key: [] for key in categories}

    if policy == Policy.exhaustive:
        sorted_results["Miscellaneous"] = []

    with Progress(
        TextColumn("[bold blue]Processing"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Sorting", total=len(batch))
        for text in batch:
            sort_data = _gen_sort(text, categories, description)
            high_keys = [
                key for key in categories if sort_data[key] == ConfidenceLevel.high
            ]
            if policy == Policy.fuzzy:
                for key in high_keys:
                    sorted_results[key].append(text)
            elif policy == Policy.exhaustive:
                if high_keys:
                    sorted_results[choice(high_keys)].append(text)
                else:
                    sorted_results["Miscellaneous"].append(text)
            progress.update(task, advance=1)
    return sorted_results
