from enum import Enum
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, create_model
from random import choice, shuffle
from time import sleep, time
from typing import List, Dict, Optional, Set
from tiktoken import encoding_for_model, get_encoding
from datetime import datetime
import json


class OutputFormat(BaseModel):
    output: str


class CategoryModel(BaseModel):
    categories: List[str]


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class CombineDecision(BaseModel):
    decision: ConfidenceLevel


DEFAULT_MODEL = "gpt-5.6-luna"
PRICING_TABLE = {
    "gpt-5.6-luna": {
        "context_window": 1_050_000,
        "input_per_1m": 1.00,
        "output_per_1m": 6.00,
        "encoding": "o200k_base",
    },
    "gpt-4o": {
        "context_window": 200000,
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "encoding": "o200k_base",
    },
    "gpt-4o-mini": {
        "context_window": 200000,
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "encoding": "o200k_base",
    },
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "input_per_1m": 3.00,
        "output_per_1m": 6.00,
        "encoding": "cl100k_base",
    },
}

_client = OpenAI()
_MODEL = DEFAULT_MODEL
_model_info_cache = {}
_encoder = get_encoding("cl100k_base")
_cost_tracker = 0.0
MAX_RETRIES = 10


def _resolve_model_info(model: str) -> Dict[str, float]:
    info = PRICING_TABLE.get(
        model,
        {
            "context_window": 100000,
            "input_per_1m": 0.50,
            "output_per_1m": 1.50,
            "encoding": "cl100k_base",
        },
    )
    try:
        enc = encoding_for_model(model)
    except Exception:
        try:
            enc = get_encoding(info.get("encoding", "cl100k_base"))
        except Exception:
            enc = get_encoding("cl100k_base")
    return {
        "context_window": int(info["context_window"]),
        "cost_per_input_token": float(info["input_per_1m"]) / 1_000_000.0,
        "cost_per_output_token": float(info["output_per_1m"]) / 1_000_000.0,
        "encoding": enc,
    }


def _add_cost(messages, response_text: str, stats: Dict):
    global _cost_tracker
    if isinstance(messages, list):
        input_tokens = sum(len(_encoder.encode(m.get("content", ""))) for m in messages)
    else:
        input_tokens = len(_encoder.encode(messages.get("content", "")))
    output_tokens = len(_encoder.encode(response_text or ""))
    _cost_tracker += (
        input_tokens * _model_info_cache["cost_per_input_token"]
        + output_tokens * _model_info_cache["cost_per_output_token"]
    )
    stats["tokens"]["input"] += int(input_tokens)
    stats["tokens"]["output"] += int(output_tokens)


def _select_corpus(
    batch: List[str], randomize: bool = True, ratio: float = 0.4
) -> List[str]:
    texts = [t for t in batch if isinstance(t, str) and t.strip()]
    if randomize:
        shuffle(texts)
    limit = int(_model_info_cache["context_window"] * ratio)
    selected = []
    count = 0
    for t in texts:
        tok = len(_encoder.encode(t))
        if tok > limit:
            continue
        if count + tok > limit:
            break
        selected.append(t)
        count += tok
    return selected if selected else texts[: max(1, min(len(texts), 50))]


def _gen_categories(
    batch: List[str],
    min_clusters: int,
    max_clusters: int,
    description: str,
    stats: Dict,
) -> List[str]:
    stats["calls"]["gen_categories"] += 1
    sample = _select_corpus(batch)
    desc = (
        f"A description of the batch is as follows: {description}"
        if description
        else ""
    )
    system_message = (
        "You are a text categorizer. "
        f"When given a batch of objects, generate between {min_clusters} and {max_clusters} distinct categories. "
        f"{desc}\n"
        "Reply in JSON with key categories and value as a list of category names."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n\n".join(sample)},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.responses.parse(
                model=_MODEL, input=messages, text_format=CategoryModel
            )
            cats = [
                c
                for c in response.output_parsed.categories
                if isinstance(c, str) and c.strip()
            ]
            if not cats:
                cats = ["General"]
            cats = list(dict.fromkeys(cats))
            _add_cost(messages, json.dumps(cats), stats)
            return cats
        except OpenAIError:
            stats["retries"] += 1
            sleep(min(30.0, 1.5**attempt))
    return ["General"]


def _create_SortModel(category_keys: List[str]) -> BaseModel:
    fields = {k: (ConfidenceLevel, ConfidenceLevel.low) for k in category_keys}
    return create_model("SortModel", **fields)


def _gen_sort(
    text: str, category_keys: List[str], description: Optional[str], stats: Dict
) -> Dict[str, ConfidenceLevel]:
    stats["calls"]["sort"] += 1
    keys = [k for k in category_keys if k != "Miscellaneous"]
    SortModel = _create_SortModel(keys)
    desc = (
        f"The categories are generated from a larger set, described as follows: {description}"
        if description
        else ""
    )
    system_message = (
        "You are a text sorter. "
        f"When given text, assign confidence for each category from this list: {', '.join(keys)}. "
        f"{desc}\n"
        "Provide JSON where keys are category names and values are one of high medium low."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.responses.parse(
                model=_MODEL, input=messages, text_format=SortModel
            )
            data = response.output_parsed.model_dump()
            _add_cost(messages, json.dumps({k: str(v) for k, v in data.items()}), stats)
            return data
        except OpenAIError:
            stats["retries"] += 1
            sleep(min(30.0, 1.5**attempt))
    return {k: ConfidenceLevel.low for k in keys}


def _gen_combined_category(high_keys: List[str], stats: Dict) -> Optional[str]:
    stats["calls"]["combine_decision"] += 1
    sys = {
        "role": "system",
        "content": "You decide whether categories should be combined. Reply as JSON with key decision and value one of high medium low.",
    }
    user_message = "Categories to evaluate for combination. " + ", ".join(high_keys)
    messages = [sys, {"role": "user", "content": user_message}]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.responses.parse(
                model=_MODEL, input=messages, text_format=CombineDecision
            )
            decision = response.output_parsed.decision
            _add_cost(messages, decision.value, stats)
            if decision == ConfidenceLevel.high:
                stats["combination_attempts"] += 1
                stats["calls"]["combine_name"] += 1
                name_messages = [
                    {
                        "role": "system",
                        "content": "Name a single concise proper case label for the combined category. Return only the name.",
                    },
                    {"role": "user", "content": "Provide the name now."},
                ]
                for attempt_name in range(1, MAX_RETRIES + 1):
                    try:
                        name_resp = _client.responses.parse(
                            model=_MODEL, input=name_messages, text_format=OutputFormat
                        )
                        name = name_resp.output_parsed.output.strip()
                        _add_cost(name_messages, name, stats)
                        if name:
                            stats["combined_merges"] += 1
                            return name
                        break
                    except OpenAIError:
                        stats["retries"] += 1
                        sleep(min(30.0, 1.5**attempt_name))
                return None
            return None
        except OpenAIError:
            stats["retries"] += 1
            sleep(min(30.0, 1.5**attempt))
    return None


def _rename_categories(
    sorted_results: Dict[str, List[str]], description: str, stats: Dict
) -> Dict[str, str]:
    stats["calls"]["rename"] += 1
    samples = {}
    for k, v in sorted_results.items():
        if k == "Miscellaneous":
            continue
        if not v:
            continue
        chosen = v[: min(20, len(v))]
        samples[k] = chosen
    if not samples:
        return {}
    prompt = {
        "task": "Rename categories to be descriptive and specific based on the provided items. Preserve semantic meaning. Return a JSON object mapping original names to new names.",
        "description": description,
        "categories": samples,
    }
    messages = [
        {"role": "system", "content": "You rename category labels."},
        {"role": "user", "content": json.dumps(prompt)},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _client.responses.parse(
                model=_MODEL, input=messages, text_format=OutputFormat
            )
            text = resp.output_parsed.output
            _add_cost(messages, text, stats)
            try:
                mapping = json.loads(text)
            except Exception:
                mapping = {}
            result = {}
            used = set()
            for old, new in mapping.items():
                if not isinstance(new, str) or not new.strip():
                    continue
                nm = new.strip()
                base = nm
                idx = 2
                while nm in used:
                    nm = f"{base} {idx}"
                    idx += 1
                used.add(nm)
                result[old] = nm
            return result
        except OpenAIError:
            stats["retries"] += 1
            sleep(min(30.0, 1.5**attempt))
    return {}


def _refine_misc(
    sorted_results: Dict[str, List[str]],
    min_clusters: int,
    max_clusters: int,
    description: str,
    stats: Dict,
):
    misc = list(sorted_results.get("Miscellaneous", []))
    if not misc:
        return
    non_empty = [
        k for k, v in sorted_results.items() if k != "Miscellaneous" and len(v) > 0
    ]
    avg = (
        0
        if not non_empty
        else sum(len(sorted_results[k]) for k in non_empty) / len(non_empty)
    )
    if len(misc) <= max(1, int(avg * 2)):
        return
    misc_cats = _gen_categories(misc, min_clusters, max_clusters, description, stats)
    combo_counts: Dict[str, int] = {}
    combo_block: Set[str] = set()
    keep_misc: List[str] = []
    processed = 0
    for t in misc:
        sort_map = _gen_sort(t, misc_cats, description, stats)
        highs = [k for k in misc_cats if sort_map.get(k) == ConfidenceLevel.high]
        if len(highs) == 0:
            keep_misc.append(t)
        elif len(highs) == 1:
            if highs[0] not in sorted_results:
                sorted_results[highs[0]] = []
            sorted_results[highs[0]].append(t)
        else:
            highs.sort()
            key = " + ".join(highs)
            combo_counts[key] = combo_counts.get(key, 0) + 1
            if combo_counts[key] > 5:
                if key not in combo_block:
                    new_cat = _gen_combined_category(highs, stats)
                    if new_cat:
                        combo_block.add(key)
                        if new_cat not in sorted_results:
                            sorted_results[new_cat] = []
                        for k in highs:
                            if k in sorted_results:
                                sorted_results[new_cat].extend(sorted_results[k])
                                sorted_results[k] = []
                        sorted_results[new_cat].append(t)
                    else:
                        combo_block.add(key)
                        sorted_results[choice(highs)].append(t)
                else:
                    sorted_results[choice(highs)].append(t)
            else:
                sorted_results[choice(highs)].append(t)
        processed += 1
    sorted_results["Miscellaneous"] = keep_misc


def assort(
    batch: List[str],
    min_clusters: int = 2,
    max_clusters: int = 5,
    description: str = "",
    model: Optional[str] = None,
    rename_final: bool = True,
) -> (Dict[str, object], Dict[str, object]):
    global _MODEL, _model_info_cache, _encoder, _cost_tracker
    start_time = time()
    _MODEL = model or DEFAULT_MODEL
    info = _resolve_model_info(_MODEL)
    _model_info_cache = {
        "context_window": info["context_window"],
        "cost_per_input_token": info["cost_per_input_token"],
        "cost_per_output_token": info["cost_per_output_token"],
    }
    _encoder = info["encoding"]
    _cost_tracker = 0.0
    stats = {
        "model": _MODEL,
        "items_total": 0,
        "initial_categories_count": 0,
        "final_categories_count": 0,
        "miscellaneous_count": 0,
        "calls": {
            "gen_categories": 0,
            "sort": 0,
            "combine_decision": 0,
            "combine_name": 0,
            "rename": 0,
        },
        "retries": 0,
        "tokens": {"input": 0, "output": 0},
        "combination_attempts": 0,
        "combined_merges": 0,
        "elapsed_seconds": 0.0,
        "cost_usd": 0.0,
        "category_sizes": {},
    }
    items = [t for t in batch if isinstance(t, str) and t.strip()]
    stats["items_total"] = len(items)
    if not items:
        results = {"sorted_results": {}}
        stats["elapsed_seconds"] = time() - start_time
        stats["cost_usd"] = _cost_tracker
        return results, stats
    initial_categories = _gen_categories(
        items, min_clusters, max_clusters, description, stats
    )
    initial_categories = [c for c in initial_categories if c != "Miscellaneous"]
    if not initial_categories:
        initial_categories = ["General"]
    stats["initial_categories_count"] = len(initial_categories)
    sorted_results: Dict[str, List[str]] = {k: [] for k in initial_categories}
    sorted_results["Miscellaneous"] = []
    combo_counts: Dict[str, int] = {}
    combo_block: Set[str] = set()
    for i, t in enumerate(items):
        cats = list(sorted_results.keys())
        if "Miscellaneous" in cats:
            cats.remove("Miscellaneous")
        sort_map = _gen_sort(t, cats, description, stats)
        highs = [k for k in cats if sort_map.get(k) == ConfidenceLevel.high]
        if len(highs) == 0:
            sorted_results["Miscellaneous"].append(t)
        elif len(highs) == 1:
            sorted_results[highs[0]].append(t)
        else:
            highs.sort()
            key = " + ".join(highs)
            combo_counts[key] = combo_counts.get(key, 0) + 1
            if combo_counts[key] > 5:
                if key not in combo_block:
                    new_cat = _gen_combined_category(highs, stats)
                    if new_cat:
                        combo_block.add(key)
                        if new_cat not in sorted_results:
                            sorted_results[new_cat] = []
                        for k in highs:
                            if k in sorted_results:
                                sorted_results[new_cat].extend(sorted_results[k])
                                sorted_results[k] = []
                        sorted_results[new_cat].append(t)
                    else:
                        combo_block.add(key)
                        sorted_results[choice(highs)].append(t)
                else:
                    sorted_results[choice(highs)].append(t)
            else:
                sorted_results[choice(highs)].append(t)
        _refine_misc(sorted_results, min_clusters, max_clusters, description, stats)
        evict = [
            key
            for key, value in sorted_results.items()
            if key != "Miscellaneous" and len(value) + 1 < 0.03 * (i + 1)
        ]
        for key in evict:
            sorted_results["Miscellaneous"].extend(sorted_results.pop(key))
    if rename_final:
        mapping = _rename_categories(sorted_results, description, stats)
        if mapping:
            new_results: Dict[str, List[str]] = {}
            for old, items_list in sorted_results.items():
                if old == "Miscellaneous":
                    new_results[old] = items_list
                    continue
                new_name = (
                    mapping.get(old, old).strip()
                    if isinstance(mapping.get(old, old), str)
                    else old
                )
                if not new_name:
                    new_name = old
                if new_name not in new_results:
                    new_results[new_name] = []
                new_results[new_name].extend(items_list)
            sorted_results = new_results
    stats["final_categories_count"] = len(
        [k for k, v in sorted_results.items() if k != "Miscellaneous"]
    )
    stats["miscellaneous_count"] = len(sorted_results.get("Miscellaneous", []))
    stats["category_sizes"] = {k: len(v) for k, v in sorted_results.items()}
    stats["elapsed_seconds"] = time() - start_time
    stats["cost_usd"] = _cost_tracker
    results = {"sorted_results": sorted_results}
    return results, stats
