from enum import Enum
from genai_prices import UpdatePrices, Usage, calc_price
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, create_model
from random import choice, shuffle
from time import sleep, time
from typing import List, Dict, Optional, Set
from tiktoken import encoding_for_model, get_encoding
from tqdm import tqdm
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

_client = None
_MODEL = DEFAULT_MODEL
_model_info_cache = {}
_encoder = get_encoding("cl100k_base")
_cost_tracker = 0.0
_price_updater = UpdatePrices()
_price_updater_started = False
MAX_RETRIES = 10


def _resolve_model_info(model: str) -> Dict[str, object]:
    provider_id, separator, model_name = model.partition(":")
    if not separator:
        model_name = provider_id
        provider_id = None
    elif provider_id == "azure_openai":
        provider_id = "azure"

    profile = getattr(_client, "profile", None) or {}
    context_window = (
        profile.get("max_input_tokens") if isinstance(profile, dict) else None
    )
    if not context_window:
        try:
            pricing = calc_price(Usage(), model_name, provider_id=provider_id)
            context_window = pricing.model.context_window
        except (LookupError, ValueError):
            pass

    try:
        enc = encoding_for_model(model_name)
    except Exception:
        is_openai_model = provider_id == "openai" or (
            provider_id is None and model_name.startswith(("gpt-", "o"))
        )
        enc = get_encoding("o200k_base" if is_openai_model else "cl100k_base")

    return {
        "provider_id": provider_id,
        "model_name": model_name,
        "context_window": int(context_window or 100000),
        "encoding": enc,
    }


def _add_cost(response, messages, response_text: str, stats: Dict):
    global _cost_tracker
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    input_details = usage_metadata.get("input_token_details", {})
    output_details = usage_metadata.get("output_token_details", {})

    if usage_metadata:
        input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
    else:
        if isinstance(messages, list):
            input_tokens = sum(
                len(_encoder.encode(m.get("content", ""))) for m in messages
            )
        else:
            input_tokens = len(_encoder.encode(messages.get("content", "")))
        output_tokens = len(_encoder.encode(response_text or ""))

    stats["tokens"]["input"] += int(input_tokens)
    stats["tokens"]["output"] += int(output_tokens)

    if _cost_tracker is None:
        return

    response_metadata = getattr(response, "response_metadata", None) or {}
    model_name = (
        usage_metadata.get("model_name")
        or response_metadata.get("model_name")
        or response_metadata.get("model")
        or _model_info_cache["model_name"]
    )
    usage = Usage(
        input_tokens=input_tokens,
        cache_write_tokens=int(input_details.get("cache_creation", 0) or 0),
        cache_read_tokens=int(input_details.get("cache_read", 0) or 0),
        output_tokens=output_tokens,
        input_audio_tokens=int(input_details.get("audio", 0) or 0),
        output_audio_tokens=int(output_details.get("audio", 0) or 0),
    )
    try:
        price = calc_price(
            usage,
            model_name,
            provider_id=_model_info_cache["provider_id"],
        )
    except (LookupError, ValueError):
        _cost_tracker = None
    else:
        _cost_tracker += float(price.total_price)


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
            response = _client.with_structured_output(
                CategoryModel, include_raw=True
            ).invoke(messages)
            cats = [
                c
                for c in response["parsed"].categories
                if isinstance(c, str) and c.strip()
            ]
            if not cats:
                cats = ["General"]
            cats = list(dict.fromkeys(cats))
            _add_cost(response["raw"], messages, json.dumps(cats), stats)
            return cats
        except Exception:
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
            response = _client.with_structured_output(
                SortModel, include_raw=True
            ).invoke(messages)
            data = response["parsed"].model_dump()
            _add_cost(
                response["raw"],
                messages,
                json.dumps({k: str(v) for k, v in data.items()}),
                stats,
            )
            return data
        except Exception:
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
            response = _client.with_structured_output(
                CombineDecision, include_raw=True
            ).invoke(messages)
            decision = response["parsed"].decision
            _add_cost(response["raw"], messages, decision.value, stats)
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
                        name_resp = _client.with_structured_output(
                            OutputFormat, include_raw=True
                        ).invoke(name_messages)
                        name = name_resp["parsed"].output.strip()
                        _add_cost(name_resp["raw"], name_messages, name, stats)
                        if name:
                            stats["combined_merges"] += 1
                            return name
                        break
                    except Exception:
                        stats["retries"] += 1
                        sleep(min(30.0, 1.5**attempt_name))
                return None
            return None
        except Exception:
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
            response = _client.with_structured_output(
                OutputFormat, include_raw=True
            ).invoke(messages)
            text = response["parsed"].output
            _add_cost(response["raw"], messages, text, stats)
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
        except Exception:
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
    show_progress: bool = False,
) -> (Dict[str, object], Dict[str, object]):
    global _client, _MODEL, _model_info_cache, _encoder, _cost_tracker
    global _price_updater_started
    start_time = time()
    _MODEL = model or DEFAULT_MODEL
    _client = init_chat_model(_MODEL)
    if not _price_updater_started:
        try:
            _price_updater.start(wait=5)
        except Exception:
            pass
        _price_updater_started = True
    info = _resolve_model_info(_MODEL)
    _model_info_cache = {
        "provider_id": info["provider_id"],
        "model_name": info["model_name"],
        "context_window": info["context_window"],
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
    for i, t in enumerate(
        tqdm(items, desc="Assorting", disable=not show_progress)
    ):
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
