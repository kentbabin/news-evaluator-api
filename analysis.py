import utils, json, asyncio, scraper, models, evaluate, db, os, random
from typing import AsyncGenerator, List, Any, Optional, Dict
from fastapi import HTTPException
from pydantic import BaseModel, HttpUrl

class Summary(BaseModel):
    summary: str
    topics: List[str]
    type: str

class SummaryResponse(BaseModel):
    summary: str
    topics: List[str]
    type: str
    model: str

class Article(BaseModel):
    perspective: str
    tone_language: Optional[List[str]]
    fairness: str
    headline_article: str
    notes: Optional[str] = None

class Publication(BaseModel):
    source_of_funding: Optional[List[str]]
    location: Optional[str]
    ownership: Optional[str]

class Evaluation(BaseModel):
    article: Article
    publication: Publication

class ModelEvaluation(BaseModel):
    model: str
    article: Article
    publication: Publication
    raw: Optional[Dict[str, Any]] = None

class EvaluationDisagreement(BaseModel):
    model: Optional[str]
    value: Optional[Any]

class Disagreement(BaseModel):
    field: Optional[str]
    evaluations: Optional[List[EvaluationDisagreement]]

class Consensus(BaseModel):
    article: Article
    publication: Publication
    confidence: float
    disagreements: Optional[List[Disagreement]] = []
    notes: str

class ConsensusResponse(BaseModel):
    article: Article
    publication: Publication
    confidence: float
    disagreements: Optional[List[Disagreement]] = []
    notes: str
    model: str

class AnalyzeResponse(BaseModel):
    url: HttpUrl
    title: Optional[str]
    authors: Optional[List[str]]
    publication: Optional[str]
    published_at: Optional[str]
    summary: SummaryResponse
    evaluations: List[ModelEvaluation]
    consensus: ConsensusResponse
    history: Optional[Dict[str, Any]]

# ---- Shared function ----
async def run_analysis(req, stream: bool = False) -> dict | AsyncGenerator[str, None]:
    url = str(req.url)

    # --- 1. Quick pre-check ---
    if not utils.looks_like_article_url(url):
        raise HTTPException(status_code=400, detail="This link doesn't appear to be a news article URL.")

    # --- 2. Scrape article ---
    try:
        scraped = await scraper.scrape_article(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {e}")

    text = scraped.get("text") or ""
    title = scraped.get("title") or ""
    publication = scraped.get("publication") or ""

    if len(text.strip()) < 500:
        raise HTTPException(status_code=400, detail="Extracted content is too short — likely not a full article.")
    if not title or len(title.strip()) < 5:
        raise HTTPException(status_code=400, detail="No valid title detected — likely not a news article.")

    metadata = {
        "title": title,
        "authors": scraped.get("authors"),
        "publication": publication,
        "published_at": scraped.get("published_at"),
        "url": url,
        "content": text[:req.max_summary_chars],
    }

    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL")

    SUMMARY_PROMPT = '''
    You are a news analyst. Analyze the provided news article and produce a structured JSON object with three fields: summary, topics, and type.

    Instructions:

    - summary: Write a 2-4 sentence summary capturing the key facts or conclusions of the article. Use neutral and concise language.
    - topics: Provide a list of 1-3 broad topics (e.g., “Climate Change”, “Elections”, “Technology”). Each topic must start with a capital letter.
    - type: Classify the article as either "Opinion" or "Reporting".
    - Opinion = subjective commentary, argument, or editorial tone.
    - Reporting = fact-based news, analysis, or investigative content.

    Output format (JSON only):

    {
        "summary": "",
        "topics": [],
        "type": ""
    }

    Return only valid JSON — no markdown, explanations, or extra text.  No cleaning should be required (i.e., no "```json" at the start and "```" at the end).
    '''

    p = SUMMARY_PROMPT + "\n---\n" + json.dumps({
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "publication": metadata.get("publication"),
            "url": metadata.get("url"),
            "content_snippet": metadata.get("content")[:req.max_summary_chars]
        }, ensure_ascii=False, indent=2)

    # result = await models.call_ollama(p, Summary.model_json_schema(), SUMMARY_MODEL)
    result = await models.call_openrouter(p, Summary.model_json_schema(), json.loads(SUMMARY_MODEL))
    text = Summary.model_validate_json(result['text'])

    # If LLM returned a structured model, convert it cleanly
    if hasattr(text, "model_dump"):
        parsed = text.model_dump()
    elif isinstance(text, dict):
        parsed = text
    else:
        # fallback if something goes wrong
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}

    summary_response = SummaryResponse(
        summary=parsed.get("summary", {}),
        topics=parsed.get("topics", {}),
        type=parsed.get("type", {}),
        model=result["model"]
    )

    # --- 3. Make evaluations ---
    prompt = evaluate.make_evaluation_prompt(metadata)

    model_list = json.loads(os.getenv("EVALUATION_MODELS"))

    random.shuffle(model_list)
    pairs = [(model_list[i], model_list[i + 1]) for i in range(0, min(len(model_list), 6), 2)]
    pairs = pairs[:3]

    calls = []

    for pair in pairs:
        calls.append(models.call_openrouter(prompt, Evaluation.model_json_schema(), model=pair))

    # --- Streaming branch ---
    async def event_stream():
        yield utils.sse_event("status", {"message": "Evaluating article..."})
        raw_results = await asyncio.gather(*calls, return_exceptions=True)

        evaluations = []
        for r in raw_results:
            if isinstance(r, Exception):
                ev = {
                    "model": "error",
                    "article": {
                        "bias": "Unknown",
                        "credibility": "Unknown",
                        "notes": f"Model call failed: {r}",
                    },
                    "publication": {"source_of_funding": None, "location": None},
                    "raw": None,
                }
                evaluations.append(ev)
                yield utils.sse_event("evaluation", evaluation)
                continue

            text = r.get("text")

            # If LLM returned a structured model, convert it cleanly
            if hasattr(text, "model_dump"):
                parsed = text.model_dump()
            elif isinstance(text, dict):
                parsed = text
            else:
                # fallback if something goes wrong
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = {}
            
            if not isinstance(parsed, dict):
                parsed = json.loads(json.dumps(parsed, default=str))

            evaluation = {
                "model": r.get("model"),
                "article": parsed.get("article", {}),
                "publication": parsed.get("publication", {}),
                "raw": {"text": parsed, "normalized": parsed},
            }
            evaluations.append(evaluation)
            yield utils.sse_event("evaluation", evaluation)

        yield utils.sse_event("status", {"message": "Finding consensus..."})
        consensus = await evaluate.aggregate_evaluations(evaluations, metadata)

        yield utils.sse_event("status", {"message": "Getting historical data..."})
        history = await db.get_consensus_stats_for_url(url)

        final_result = AnalyzeResponse(
            url = url,
            title = metadata.get("title"),
            authors = metadata.get("authors"),
            publication = metadata.get("publication"),
            published_at = str(metadata.get("published_at")),
            summary=summary_response,
            evaluations = evaluations,
            consensus = consensus,
            history = history,
        )
        
        db.insert_result(url, metadata.get("publication"), final_result.model_dump_json())

        yield utils.sse_event("done", final_result.model_dump())

    if stream:
        return event_stream()

    # --- Non-streaming branch ---
    raw_results = await asyncio.gather(*calls, return_exceptions=True)
    evaluations = []
    for r in raw_results:
        if isinstance(r, Exception):
            evaluations.append({
                "model": "error",
                "article": {"bias": "Unknown", "credibility": "Unknown", "notes": f"Model call failed: {r}"},
                "publication": {"source_of_funding": None, "location": None},
                "raw": None,
            })
            continue

        text = r.get("text")

        # If LLM returned a structured model, convert it cleanly
        if hasattr(text, "model_dump"):
            parsed = text.model_dump()
        elif isinstance(text, dict):
            parsed = text
        else:
            # fallback if something goes wrong
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {}

        evaluations.append({
            "model": r.get("model"),
            "article": parsed.get("article", {}),
            "publication": parsed.get("publication", {}),
            "raw": {"text": parsed, "normalized": parsed},
        })

    consensus = await evaluate.aggregate_evaluations(evaluations, metadata)

    history = await db.get_consensus_stats_for_url(url)

    final_result = AnalyzeResponse(
            url = url,
            title = metadata.get("title"),
            authors = metadata.get("authors"),
            publication = metadata.get("publication"),
            published_at = str(metadata.get("published_at")),
            summary = summary_response,
            evaluations = evaluations,
            consensus = consensus,
            history = history,
        )
    
    db.insert_result(url, metadata.get("publication"), final_result.model_dump_json())

    return final_result