import duckdb, os, json, models, utils
from collections import Counter
from typing import List, Dict, Any
from pydantic import BaseModel, RootModel

class StatsAnswer(BaseModel):
    answer: str
    count: int

class FieldStats(RootModel[Dict[str, List[StatsAnswer]]]):
    pass

class Deduplication(BaseModel):
    stats: List[FieldStats]
    

CONSENSUS_FIELDS = [
    "perspective",
    "tone_language",
    "fairness",
    "headline_article",
    "source_of_funding",
    "ownership",
    "location",
]

DB_PATH = "file.db"

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL")

# con = duckdb.connect("file.db")
# con.sql("CREATE SEQUENCE id_seq START 1")
# con.sql(
#     "CREATE TABLE results (id INTEGER PRIMARY KEY DEFAULT nextval('id_seq'), url TEXT, publication TEXT, result JSON, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
#     )

# con.sql("CREATE SEQUENCE articles_id_seq START 1")
# con.sql(
#     "CREATE TABLE articles (id INTEGER PRIMARY KEY DEFAULT nextval('articles_id_seq'), url TEXT, result JSON, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
#     )

def insert_result(url: str, publication: str, result: str):
    con = duckdb.connect(DB_PATH)
    con.sql(
        'INSERT INTO results (url, publication, result) VALUES ($url, $publication, $result)',
        params={'url': url, 'publication': publication, 'result': result}
    )
    con.close()

async def get_consensus_stats_for_url(url: str):
    con = duckdb.connect(DB_PATH)
    rows = con.execute("SELECT result FROM results WHERE url = ?", [url]).fetchall()
    con.close()
    if not rows:
        return {"url": url, "stats": {}}

    field_values = {field: [] for field in CONSENSUS_FIELDS}
    field_answers = {field: [] for field in CONSENSUS_FIELDS}

    for (result_json,) in rows:
        try:
            data = json.loads(result_json)
            consensus = data.get("consensus", {})
            article = consensus.get("article", {})
            publication = consensus.get("publication", {})

            mapping = {
                "perspective": article.get("perspective"),
                "tone_language": article.get("tone_language"),
                "fairness": article.get("fairness"),
                "headline_article": article.get("headline_article"),
                "source_of_funding": publication.get("source_of_funding"),
                "ownership": publication.get("ownership"),
                "location": publication.get("location"),
            }

            # Track consensus values
            for field, value in mapping.items():
                if value is not None:
                    field_values[field].append(value)

            for ev in data.get("evaluations", []):
                article = ev.get("article", {})
                publication = ev.get("publication", {})
                mapping = {
                    "perspective": article.get("perspective"),
                    "tone_language": article.get("tone_language"),
                    "fairness": article.get("fairness"),
                    "headline_article": article.get("headline_article"),
                    "source_of_funding": publication.get("source_of_funding"),
                    "ownership": publication.get("ownership"),
                    "location": publication.get("location"),
                }
                for field, answer in mapping.items():
                    if answer is None:
                        continue
                    if isinstance(answer, list):
                        field_answers[field].extend(answer)
                    else:
                        field_answers[field].append(answer)

        except Exception as e:
            print(f"⚠️ Failed to parse record: {e}")

    # ---- Compute stats
    stats = {}
    for field, values in field_values.items():
        if not values:
            continue

        counts = Counter(["no" if str(v).lower() == "no consensus" else "yes" for v in values])
        total = sum(counts.values())
        no_ratio = (counts["no"] / total) * 100
        yes_ratio = (counts["yes"] / total) * 100

        answer_counts = Counter(field_answers[field])
        answers_list = [{"answer": ans, "count": count} for ans, count in answer_counts.items()]

        stats[field] = {
            "no_consensus": round(no_ratio, 1),
            "consensus": round(yes_ratio, 1),
            "total": total,
            "answers": answers_list,
        }

    # ---- Deduplication step (single LLM call)
    DEDUPLICATION_PROMPT = '''
    You are a text analysis system.

    The provided stats object contains multiple fields: perspective, tone_language, fairness, headline_article, source_of_funding, ownership, and location.

    Each field includes an answers list of text strings (and their counts). Your task is to group semantically or stylistically equivalent answers within each field and produce canonical labels with combined counts.

    Grouping Rules

    1. Normalize for comparison:
    - Convert to lowercase
    - Trim whitespace
    - Remove punctuation except internal hyphens (-)
    - Collapse multiple spaces into one
    2. Merge answers if they are effectively equivalent — for example:
    - Same meaning with minor wording differences (e.g., "Pro-Palestinian" ≈ "Pro Palestinian self-determination")
    - Differ only in capitalization or punctuation (e.g., "Government of Qatar" ≈ "Government Of Qatar")
    3. When merging, use the most representative or commonly occurring form as the canonical label (capitalize each major word).
    4. The count for a canonical label is the sum of counts from all merged variants.
    5. Preserve the order of fields as listed above.
    6. Return "Unknown" if a field has no valid answers.

    Output Schema

    Return only a valid JSON object in this exact structure (no markdown, code fences, or explanations):

    {
        "stats": [
            { "perspective": [ { "answer": "", "count": 0 } ] },
            { "tone_language": [ { "answer": "", "count": 0 } ] },
            { "fairness": [ { "answer": "", "count": 0 } ] },
            { "headline_article": [ { "answer": "", "count": 0 } ] },
            { "source_of_funding": [ { "answer": "", "count": 0 } ] },
            { "ownership": [ { "answer": "", "count": 0 } ] },
            { "location": [ { "answer": "", "count": 0 } ] }
        ]
    }

    Each list can contain multiple { "answer": "<canonical label>", "count": <combined count> } objects.

    Return only this JSON — no extra text or commentary. No cleaning should be required (i.e., no "```json" at the start and "```" at the end).
    '''

    prompt = DEDUPLICATION_PROMPT + "\n---\nAnswers:\n" + json.dumps(stats, indent=2)

    result = await models.call_openrouter(prompt, Deduplication.model_json_schema(), json.loads(SUMMARY_MODEL))

    text = Deduplication.model_validate_json(result["text"])
    parsed = text.model_dump()

    # flatten the nested structure
    stats_data = parsed.get("stats", [])
    deduped_map = {k: v for field_obj in stats_data for k, v in field_obj.items()}

    # ---- Merge deduplicated answers back
    for field, deduped_answers in deduped_map.items():
        if field in stats:
            total = sum(a["count"] for a in deduped_answers)
            stats[field].update({
                "total": total,
                "answers": deduped_answers,
            })

    return {"url": url, "stats": stats, "model": result["model"]} 

# --- Utility to run query --- #
def chart_query(query: str) -> List[Dict[str, Any]]:
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        conn.execute(query)
        rows = conn.fetchall()
        cols = [desc[0] for desc in conn.description]
        return [dict(zip(cols, row)) for row in rows]


# --- Format results into [{x, y:[{key,count}]}] --- #
def transform_for_chart(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms query rows into the format expected by the frontend:
    [
        {"x": "Low", "y": [{"key": "Model A", "count": 10}, ...]},
        {"x": "Medium", "y": [...]},
        ...
    ]

    Automatically detects which field to use as the 'x' value
    (e.g., 'fairness', 'headline_article', etc.).
    """
    if not rows:
        return []

    # Detect the x-axis field (first key not named 'key' or 'count')
    sample = rows[0]
    x_field = next((k for k in sample.keys() if k not in {"key", "count"}), None)
    if not x_field:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        x_value = row[x_field]
        key = row["key"]
        count = row["count"]
        grouped.setdefault(x_value, []).append({"key": key, "count": count})

    # Return standardized structure
    return [{"x": x, "y": ylist} for x, ylist in grouped.items()]
