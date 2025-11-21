import json, re
from typing import Any
from urllib.parse import urlparse

def clean_llm_json(text: str):
    """
    Safely parse JSON (object or array) from LLM output.
    Handles code fences, stray quotes, or stringified JSON.
    Returns dict, list, or {} if parsing fails.
    """
    if not text:
        return {}

    # Trim whitespace and remove common formatting wrappers
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip().strip('"').strip("'")

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try if it's double-encoded (stringified JSON inside a JSON string)
    try:
        parsed = json.loads(json.loads(text))
        return parsed
    except Exception:
        pass

    # Regex fallback to extract first valid JSON array/object
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate.replace("'", '"'))
            except Exception:
                return {}

    return {}

def normalize_evaluation(parsed: dict) -> dict:
    article = parsed.get("article", {})
    publication = parsed.get("publication", {})

    def norm_field(d: dict, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    pub_funding = norm_field(publication, "source_of_funding", "source of funding", "funding_source", "funding", "publication_funding")
    pub_location = norm_field(publication, "location", "country", "publication_location")
    pub_ownership = norm_field(publication, "ownership", "owner", "publication_ownership")

    return {
        "article": {
            "perspective": norm_field(article, "perspective", "slant", "article_perspective") or "Unknown",
            "tone_language": norm_field(article, "tone_language", "tone and language", "article_tone_language") or "Unknown",
            "fairness": norm_field(article, "fairness", "article_fairness") or "Unknown",
            "headline_article": norm_field(article, "headline_article", "article_headline_article") or "Unknown",
            "notes": article.get("notes"),
        },
        "publication": {
            "source_of_funding": pub_funding or "Unknown",
            "location": pub_location or "Unknown",
            "ownership": pub_ownership or "Unknown",
        },
    }

def extract_json_from_text(text: str) -> Any:
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None
        
def looks_like_article_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Reject bare domains or short paths
    if not path or path == "/" or len(path.strip("/").split("/")) <= 1:
        return False

    # Common article URL patterns
    if re.search(r"/\d{4}/\d{1,2}/\d{1,2}/", path):
        return True
    if any(seg in path for seg in ["article", "news", "story", "posts", "politics", "world", "economy", "health"]):
        return True
    if path.endswith(".html") or path.endswith(".htm"):
        return True

    return False

def sse_event(event: str, data):
    """Format a dict or string as an SSE event (robust JSON handling)."""
    def default(o):
        # Handle common non-serializable types
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return list(o)
        return str(o)

    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, default=default)
    return f"event: {event}\ndata: {data}\n\n"