import models, utils, json, os, analysis
from typing import List, Dict, Any
from dotenv import load_dotenv


load_dotenv()

CONSENSUS_MODEL = os.getenv("CONSENSUS_MODEL")

EVALUATION_RUBRIC = '''
You are an expert news analyst. Your goal is to help users understand the context, bias, and perspective of a news article — not to verify truth or accuracy. Provide a balanced, analytical assessment in structured JSON format.

Instructions

For the article, analyze:

- perspective: Identify the stance or slant of the article (e.g., "Pro Gun Control", "Anti Immigration", "Neutral"). Return only one value.
- tone_language: A list of adjectives describing the tone and language used (e.g., "Emotionally Charged", "Combative", "Restrained"). Return an empty list if none apply.
- fairness: Rate how well the article presents multiple viewpoints. Choose one of: "Low", "Medium", "High".
- headline_article: Evaluate the gap between headline and content. Choose one of: "Large", "Medium", "Small".
- notes: Briefly explain your reasoning for the above (1-3 sentences, concise and neutral).

For the publication, identify:

- source_of_funding: A list of known funding sources (Examples include things like advertising, government grants, reader subscriptions). If unknown, return "Unknown".
- location: The country where the publication is headquartered or registered. If unknown, return "Unknown".
- ownership: The entity or individual that owns the publication. If unknown, return "Unknown".

Formatting Rules

- Return only valid JSON, no markdown, code blocks, or explanations.
- Capitalize the first letter of every string and each item in lists.
- Ensure all fields are filled, using "Unknown" or an empty list [] where applicable.

Output Schema

{
  "article": {
    "perspective": "",
    "tone_language": [],
    "fairness": "",
    "headline_article": "",
    "notes": ""
  },
  "publication": {
    "source_of_funding": [],
    "location": "",
    "ownership": ""
  }
}

Return only this JSON structure with completed values. No cleaning should be required (i.e., no "```json" at the start and "```" at the end).
'''

def make_evaluation_prompt(metadata: Dict[str, Any]) -> str:
    return EVALUATION_RUBRIC + "\n---\n" + json.dumps({
        "title": metadata.get("title"),
        "authors": metadata.get("authors"),
        "publication": metadata.get("publication"),
        "published_at": str(metadata.get("published_at")),
        "url": metadata.get("url"),
        "content_snippet": metadata.get("content")[:3000]
    }, ensure_ascii=False, indent=2)

CONSENSUS_PROMPT = '''
You are a meta-reviewer of structured news-article evaluations produced by multiple models. Your job is to detect consensus for each field and return a single JSON object with the consensus results, a confidence score (0-1), and any disagreements.

Input: A JSON array named evaluations. Each element is an object containing a model string and two nested objects: article and publication. Example element:

{
  "model": "modelA",
  "article": {
    "perspective": "Pro-Palestine",
    "tone_language": ["Emotionally Charged", "Rousing"],
    "fairness": "Low",
    "headline_article": "Large"
  },
  "publication": {
    "source_of_funding": ["Advertising", "Reader Subscriptions"],
    "location": "United States",
    "ownership": "MediaCorp LLC"
  }
}

Rules / Normalization

1. Normalize all candidate strings by: trimming whitespace, converting to lowercase, removing punctuation except internal hyphens, and replacing multiple spaces with single spaces. Use this normalized form only for comparison; return original-cased forms in results (capitalized as described below).
2. For perspective and single-choice fields, group answers that are semantically equivalent (e.g., pro-palestinian, pro palestine, pro palestinian self-determination) by normalized token overlap. Treat as matching if normalized token overlap ≥ 0.6 or if one normalized answer is a substring of another. Also consider acronyms and their full form (e.g., British Broadcasting Corporation and BBC) to be equivalent.
3. For list fields (tone_language, source_of_funding), normalize each list item with rule (1). An item counts toward consensus only if that normalized item appears in every evaluator's list (presence in all lists required).
4. If a field cannot be determined or is missing from a model, treat that model as providing the value "unknown" for comparison purposes.
5. Always capitalize the first letter of returned strings and list items (e.g., "Pro-Palestine", ["Emotionally Charged"]). Use "No Consensus" (title case) when consensus is not reached for a field.
6. All fields must be present in the output. Use "Unknown" if no useful information exists.

Consensus Definition

- A field has consensus when all normalized evaluator answers are considered matching under the rules above.
- For list fields: consensus is the set of items present in every evaluator's normalized list. If that set is empty, return [].

Confidence (0-1)

Calculate confidence as the average of per-field agreement ratios:

- For single-choice fields: agreement ratio = (count of evaluators whose normalized value matches the consensus value) / (total evaluators). If "No Consensus", ratio = 0.
- For list fields: agreement ratio = (average, across consensus items, of fraction of evaluators that included that item). If no consensus items, ratio = 0.
- Final confidence = mean of the agreement ratios for the 7 monitored fields (article: perspective, fairness, tone_language, headline_article; publication: source_of_funding, ownership, location). Round to two decimal places.

Disagreements

- If consensus can't be reached for any field, include them in disagreements with the field name and the list of each model's provided (original) value.
- The structure for the disagreements list item is as follows:
    {
        "field": "",
        "evaluations": [
            {
                "model": "",
                "value": ""
            }
        ]
	}
- If there are no disagreements, return an empty list.

Output JSON schema (return only valid JSON — no markdown or extra text):

{
  "article": {
    "perspective": "",
    "tone_language": [],
    "fairness": "",
    "headline_article": "",
    "notes": ""
  },
  "publication": {
    "source_of_funding": [],
    "location": "",
    "ownership": ""
  },
  "confidence": 0.00,
  "disagreements": [],
  "notes": ""
}

Notes fields:

- article.notes: 1-2 concise sentences explaining the rationale behind the article-level consensus decisions (e.g., token overlap or repeated items).
- top-level notes: 1-2 concise sentences describing how you computed the overall confidence (no field-by-field detail unless necessary).

Return only the JSON object described above with filled fields. No cleaning should be required. Don't put ```json at the start and ``` at the end of the object).
''' 

async def aggregate_evaluations(evals: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"article": {"title": metadata.get('title'), "url": metadata.get('url')}, "evaluations": evals}
    prompt = CONSENSUS_PROMPT + "\n---\nInput evaluations:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    
    resp = await models.call_openrouter(prompt, analysis.Consensus.model_json_schema(), model=json.loads(CONSENSUS_MODEL))

    raw_text = resp.get("text", "")
    cleaned_text = utils.clean_llm_json(raw_text)

    if isinstance(cleaned_text, (dict, list)):
        parsed_json = cleaned_text
    else:
        # Try one more time if it’s still a string (some models double-wrap)
        try:
            parsed_json = json.loads(cleaned_text)
        except Exception:
            parsed_json = {}

    try:
        text = analysis.Consensus.model_validate(parsed_json)
    except Exception as e:
        print(f"[WARN] Consensus validation failed, using fallback. Error: {e}")
        text = analysis.Consensus.model_validate(
            {"summary": "", "conclusion": "Validation failed", "notes": str(e)}
        )

    parsed = text.model_dump() if hasattr(text, "model_dump") else text

    # text = analysis.Consensus.model_validate_json(resp['text'])

    # # If LLM returned a structured model, convert it cleanly
    # if hasattr(text, "model_dump"):
    #     parsed = text.model_dump()
    # elif isinstance(text, dict):
    #     parsed = text
    # else:
    #     # fallback if something goes wrong
    #     try:
    #         parsed = json.loads(text)
    #     except Exception:
    #         parsed = {}
    
    if parsed is None:
        return {
            "article_bias": evals[0].get('bias') if evals else 'Unknown',
            "article_credibility": evals[0].get('credibility') if evals else 'Unknown',
            "confidence": "Meta-LLM failed to return result",
            "disagreements": [],
        }
    
    consensus = analysis.ConsensusResponse(
        article=parsed.get("article", {}),
        publication=parsed.get("publication", {}),
        confidence=parsed.get("confidence", {}),
        notes=parsed.get("notes", {}),
        disagreements=parsed.get("disagreements", {}),
        model = resp["model"],
    )

    return consensus