from enum import Enum
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, create_model
from random import choice, shuffle
from requests import get
from tiktoken import encoding_for_model
from time import sleep
from typing import List, Dict
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class OutputFormat(BaseModel):
    output: str


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


_model_info_cache = _get_model_info(_MODEL)
_encoder = encoding_for_model(_MODEL)
_cost_tracker = 0.0


def _estimate_cost(batch: List[str], policy: Policy) -> float:
    calls = 1 + len(batch)
    if policy == Policy.exhaustive:
        calls += len(batch)
    input_tokens = len(_encoder.encode("\n\n".join(batch))) + sum(
        len(_encoder.encode(t)) for t in batch
    )
    if policy == Policy.exhaustive:
        input_tokens += sum(len(_encoder.encode(t)) for t in batch)
    output_tokens = calls * 50
    return (
        input_tokens * _model_info_cache["cost_per_input_token"]
        + output_tokens * _model_info_cache["cost_per_output_token"]
    )


def _add_cost(messages, response_text):
    global _cost_tracker
    if isinstance(messages, list):
        input_tokens = sum(len(_encoder.encode(m["content"])) for m in messages)
    else:
        input_tokens = len(_encoder.encode(messages["content"]))
    output_tokens = len(_encoder.encode(response_text))
    _cost_tracker += (
        input_tokens * _model_info_cache["cost_per_input_token"]
        + output_tokens * _model_info_cache["cost_per_output_token"]
    )


def _clean_batch(
    batch: List[str], randomize: bool = True, summarize: bool = False
) -> List[str]:
    if randomize:
        shuffle(batch)

    model_info = _get_model_info(_MODEL)
    encoder = encoding_for_model(_MODEL)
    context_window = model_info["context_window"]
    max_tokens = context_window / 2

    cleaned_batch = []
    current_batch = []
    token_count = 0

    for text in batch:
        if summarize:
            system_message = (
                "You are a text summarizer. "
                "When given a piece of text, "
                "you are to summarize it in a concise manner. "
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text},
            ]

            while True:
                try:
                    response = _client.responses.parse(
                        model=_MODEL, input=messages, text_format=OutputFormat
                    )
                    text = response.output_parsed.output
                    _add_cost(messages, text)
                    break

                except OpenAIError as e:
                    if "rate limit" in str(e).lower():
                        sleep(60)
                    elif "insufficient_quota" in str(e).lower():
                        print(
                            "Account is not funded, check billing at https://platform.openai.com/settings/organization/billing/"
                        )
                        exit()

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
        cleaned_batch.extend(current_batch)

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
            _add_cost(messages, str(response.output_parsed.categories))
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
    if "Miscellaneous" in category_keys:
        category_keys.remove("Miscellaneous")
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
            _add_cost(messages, str(response.output_parsed.model_dump()))
            return response.output_parsed.model_dump()
        except OpenAIError as e:
            if "rate limit" in str(e).lower():
                sleep(60)
            elif "insufficient_quota" in str(e).lower():
                print(
                    "Account is not funded, check billing at https://platform.openai.com/settings/organization/billing/"
                )
                exit()


def _gen_combined_category(high_keys: List[str]) -> str:
    user_message = (
        "Given the following categories, respond with a confidence level "
        + "of 'high', 'medium', or 'low' on whether they should be combined: "
        + f"{', '.join(high_keys)}. "
    )

    while True:
        try:
            response = _client.responses.parse(
                model=_MODEL,
                input={"role": "user", "content": user_message},
                text_format=ConfidenceLevel,
            )
            _add_cost({"role": "user", "content": user_message}, response.output_parsed)

            if response.output_parsed in ConfidenceLevel.high:
                user_message = (
                    "The categories should be combined. "
                    "Please provide a single name for the combined category."
                    "The new category name should be in proper case. "
                    "Do not add asterisks or any other special characters. "
                    "Do not include any other information in your response."
                )
                response = _client.responses.parse(
                    model=_MODEL,
                    input={"role": "user", "content": user_message},
                    text_format=OutputFormat,
                )
                _add_cost(
                    {"role": "user", "content": user_message},
                    response.output_parsed.output,
                )
                return response.output_parsed.output

            else:
                return None

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
    print_estimate: bool = True,
    confirm: bool = False,
) -> Dict[str, List[int]]:
    global _cost_tracker
    _cost_tracker = 0.0
    if print_estimate or confirm:
        est_cost = _estimate_cost(batch, policy)
        print(f"Estimated minimum cost: ${est_cost:.6f}")
        if confirm:
            proceed = input("Proceed? (y/n): ").lower()
            if proceed != "y":
                return {"sorted_results": {}, "cost": 0.0}
    categories = _gen_categories(batch, min_clusters, max_clusters, description)
    sorted_results = {key: [] for key in categories}

    if policy == Policy.exhaustive:
        sorted_results["Miscellaneous"] = []
        high_key_counter = {}

    with Progress(
        TextColumn("[bold blue]Processing"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Sorting", total=len(batch))
        for i, text in enumerate(batch):
            categories = list(sorted_results.keys())
            sort_data = _gen_sort(text, categories, description)
            high_keys = [
                key for key in categories if sort_data[key] == ConfidenceLevel.high
            ]
            if policy == Policy.fuzzy:
                for key in high_keys:
                    sorted_results[key].append(text)
            elif policy == Policy.exhaustive:
                if len(high_keys) == 0:
                    sorted_results["Miscellaneous"].append(text)
                elif len(high_keys) == 1:
                    sorted_results[high_keys[0]].append(text)
                else:
                    high_keys.sort()
                    high_key_concat = " + ".join(high_keys)
                    high_key_counter[high_key_concat] = (
                        high_key_counter.get(high_key_concat, 0) + 1
                    )

                    if high_key_counter[high_key_concat] > 5:
                        combined_category = _gen_combined_category(high_keys)

                        if combined_category:
                            if combined_category not in sorted_results:
                                sorted_results[combined_category] = []

                            for key in high_keys:
                                if key in sorted_results:
                                    sorted_results[combined_category].extend(
                                        sorted_results[key]
                                    )
                                    sorted_results[key] = []

                            sorted_results[combined_category].append(text)
                        else:
                            sorted_results[choice(high_keys)].append(text)

                if len(sorted_results["Miscellaneous"]) > 1:
                    fat_cats = [
                        key for key, value in sorted_results.items() if len(value) > 0
                    ]
                    fat_cats.remove("Miscellaneous")
                    if fat_cats:
                        average_cat_fat = sum(
                            len(sorted_results[key]) for key in fat_cats
                        ) / len(fat_cats)
                        if len(sorted_results["Miscellaneous"]) > average_cat_fat * 2:
                            cleaned_misc = _clean_batch(sorted_results["Miscellaneous"])
                            misc_categories = _gen_categories(
                                cleaned_misc, min_clusters, max_clusters, description
                            )
                            misc_high_key_counter = {}
                            for misc_text in cleaned_misc:
                                misc_sort_data = _gen_sort(
                                    misc_text, misc_categories, description
                                )
                                misc_high_keys = [
                                    key
                                    for key in misc_categories
                                    if misc_sort_data[key] == ConfidenceLevel.high
                                ]
                                if len(misc_high_keys) == 0:
                                    sorted_results["Miscellaneous"].append(misc_text)
                                elif len(misc_high_keys) == 1:
                                    sorted_results[misc_high_keys[0]].append(misc_text)
                                else:
                                    misc_high_keys.sort()
                                    misc_high_key_concat = " + ".join(misc_high_keys)
                                    if (
                                        misc_high_key_concat
                                        not in misc_high_key_counter
                                    ):
                                        misc_high_key_counter[misc_high_key_concat] = 0
                                    misc_high_key_counter[misc_high_key_concat] += 1
                                    if misc_high_key_counter[misc_high_key_concat] > 5:
                                        misc_combined_category = _gen_combined_category(
                                            misc_high_keys
                                        )
                                        if misc_combined_category:
                                            if (
                                                misc_combined_category
                                                not in sorted_results
                                            ):
                                                sorted_results[
                                                    misc_combined_category
                                                ] = []

                                        for key in misc_high_keys:
                                            if key in sorted_results:
                                                sorted_results[
                                                    misc_combined_category
                                                ].extend(sorted_results[key])
                                                sorted_results[key] = []

                                        sorted_results[misc_combined_category].append(
                                            misc_text
                                        )
                                    else:
                                        sorted_results[choice(misc_high_keys)].append(
                                            misc_text
                                        )

                evict = {}
                for key, value in sorted_results.items():
                    if len(value) + 1 < 0.03 * (i + 1):
                        if "Miscellaneous" not in sorted_results:
                            sorted_results["Miscellaneous"] = []
                        sorted_results["Miscellaneous"].extend(value)
                        evict[key] = value
                sorted_results = {
                    k: v for k, v in sorted_results.items() if k not in evict
                }
            progress.update(task, advance=1)
    return {"sorted_results": sorted_results, "cost": _cost_tracker}
