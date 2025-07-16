from enum import Enum
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, create_model
from time import sleep
from typing import List, Dict
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class CategoryModel(BaseModel):
    categories: List[str]


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


_MODEL = "gpt-4o-mini"
_client = OpenAI()


def _gen_categories(
    batch: List[str], min_clusters: int, max_clusters: int
) -> List[str]:
    max_clusters = max_clusters - 1

    system_message = (
        "You are a text categorizer. "
        + "When given a batch of objects, you are to come up with "
        + f"between {min_clusters} and NO MORE THAN {max_clusters} distinct categories "
        + "that the objects could be distinctly sorted into. "
        + "\n"
        + "-" * 80
        + "\n"
        + "Your reply should be in JSON format, with categories as a single key, "
        + "followed by a list of categories that the objects would best fit into. "
        + f"And remember - do not create more than {max_clusters} categories."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n\n".join(batch)},
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


def _gen_sort(text: str, category_keys: List[str]) -> str:
    SortModel = _create_SortModel(category_keys)
    system_message = (
        "You are a text sorter. "
        "When given a piece of text, you are to sort it into one of the following categories "
        f"{', '.join(category_keys)}. "
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
    batch: List[str], min_clusters: int = 2, max_clusters: int = 5
) -> Dict[str, List[int]]:
    categories = _gen_categories(batch, min_clusters, max_clusters)
    sorted_results = {key: [] for key in categories}
    with Progress(
        TextColumn("[bold blue]Processing"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Sorting", total=len(batch))
        for text in batch:
            sort_data = _gen_sort(text, categories)
            for key in categories:
                if sort_data[key] == ConfidenceLevel.high:
                    sorted_results[key].append(text)
            progress.update(task, advance=1)
    return sorted_results
