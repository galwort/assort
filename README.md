# assort

Text clustering and sorting with an LLM that discovers categories, classifies items, optionally merges overlapping themes, and cleans up labels. Designed for quick drop in use with a single function call and clear cost tracking.

## Highlights

- Discovers category names from your data
- Sorts each item with calibrated confidence per category
- Merges overlapping themes when the model judges a high likelihood of overlap
- Refines the miscellaneous bucket when it is too large
- Optionally renames categories to be clearer and more specific
- Tracks tokens and final cost in USD
- Simple one function API that returns results and rich stats

## Install

```bash
pip install assort
```

You also need an OpenAI API key available to the runtime, for example

```bash
export OPENAI_API_KEY=sk_your_key_here
```

## Quick start

```python
from assort import assort

texts = [
    "Build a responsive landing page in React",
    "How to index a Postgres table",
    "Cognitive behavioral therapy exercises",
    "Vector search with Azure AI Search",
    "Tailwind utility classes for layouts",
    "Managing anxiety before a big presentation",
]

results, stats = assort(
    texts,
    min_clusters=3,
    max_clusters=6,
    description="Short notes that mix software topics and mental health topics",
)

print(results["sorted_results"])
```

Example shape of `sorted_results`

```python
{
    "Front End Engineering": [
        "Build a responsive landing page in React",
        "Tailwind utility classes for layouts"
    ],
    "Data and Search": [
        "Vector search with Azure AI Search",
        "How to index a Postgres table"
    ],
    "Anxiety and CBT": [
        "Cognitive behavioral therapy exercises",
        "Managing anxiety before a big presentation"
    ],
    "Miscellaneous": []
}
```

## API

### assort

```python
results, stats = assort(
    batch,
    min_clusters=2,
    max_clusters=5,
    description="",
    model=None,
    rename_final=True,
    show_progress=False,
)
```

Parameters

- `batch`
  List of strings to categorize. Empty or blank strings are ignored.

- `min_clusters` and `max_clusters`
  Bounds for initial category discovery.

- `description`
  Optional corpus context. Helps the model choose better category boundaries and names.

- `model`
  Optional LangChain model name, preferably in `"provider:model"` format. If
  omitted, the existing OpenAI default is used. The selected model must support
  structured output through its LangChain integration.

- `rename_final`
  If true, the library proposes clearer category names at the end based on samples from each group.

- `show_progress`
  If true, displays progress while items are assorted.

Returns

- `results`
  Dict with key `sorted_results`. Values are lists of the original items per category. A `Miscellaneous` bucket is always present.

- `stats`
  Dict with detailed run information

  - `model`
  - `items_total`
  - `initial_categories_count`
  - `final_categories_count`
  - `miscellaneous_count`
  - `calls` with counts for internal steps
  - `retries` for API retries with backoff
  - `tokens` with `input` and `output` counts
  - `combination_attempts` and `combined_merges`
  - `elapsed_seconds`
  - `cost_usd` calculated from token counts and the internal price table
  - `category_sizes` mapping category to item count

## How it works

- Category discovery
  The model reads a sample of your corpus and proposes a set of category names between your bounds.

- Sorting
  Each item is scored for every discovered category with confidences high, medium, low. High confidence categories win. Ties are broken by simple rules.

- Combining overlapping themes
  When items frequently score high for the same pairs or sets of categories, the library asks the model if they should be combined. On a high decision, a single concise name is requested and the merge proceeds.

- Refining miscellaneous
  If `Miscellaneous` grows larger than a data guided threshold, the same discovery and sorting routine runs on that subset. Items are pulled out into new focused categories when possible.

- Renaming for clarity
  At the end, the library proposes clearer names that preserve meaning using a small sample from each category. Names are deduplicated.

## Cost and tokens

- Token accounting uses `tiktoken` with an encoder chosen for the active model.
- The final `cost_usd` is calculated from token counts and an internal price table.

## Use another model provider

Install the provider's LangChain integration, configure its API key, and pass a
`"provider:model"` string through the existing `model` argument.

```bash
pip install langchain-anthropic
export ANTHROPIC_API_KEY=your_key_here
```

```python
results, stats = assort(
    texts,
    model="anthropic:claude-sonnet-4-6",
)
```

## Advanced examples

Run and keep original names

```python
from assort import assort

texts = [...]
results, stats = assort(
    texts,
    min_clusters=4,
    max_clusters=8,
    description="Product feedback notes",
    rename_final=False,
)
```

Inspect stats for simple analytics

```python
results, stats = assort(texts)

sizes = stats["category_sizes"]
by_size = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
for name, count in by_size:
    print(name, count)
```

## Behavior notes

- Non deterministic sampling is used during corpus selection, so runs can vary.
- The module keeps a single LangChain client and encoder in module scope. In process concurrency is not recommended. Use separate processes for parallel work.
