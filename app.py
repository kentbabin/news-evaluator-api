from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
import models, analysis, db
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

app = FastAPI(title="News Evaluator Prototype")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a limiter — identify clients by IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class AnalyzeRequest(BaseModel):
    url: HttpUrl
    max_summary_chars: Optional[int] = 10000

@app.post("/analyze")
@limiter.limit("3/minute")
async def analyze(req: AnalyzeRequest, request: Request):
    """Non-streaming version"""
    return await analysis.run_analysis(req, stream=False)

@app.post("/analyze/stream")
@limiter.limit("1/minute")
async def analyze_stream(req: AnalyzeRequest, request: Request):
    """Streaming version (SSE)"""
    return StreamingResponse(
        await analysis.run_analysis(req, stream=True),
        media_type="text/event-stream"
    )

@app.get("/charts")
@limiter.limit("20/minute")
def get_charts(request: Request):
    charts = {}

    # Common grouping keys
    groupings = {
        "by_model": "json_extract_string(value, '$.model')",
        "by_publication": "publication",  # use top-level column
    }

    # Define chart categories: field name + display order
    chart_categories = {
        "fairness": {
            "json_path": "$.article.fairness",
            "order_case": """
                CASE json_extract_string(value, '$.article.fairness')
                    WHEN 'Low' THEN 1
                    WHEN 'Medium' THEN 2
                    WHEN 'High' THEN 3
                    ELSE 99
                END
            """,
        },
        "headline_article": {
            "json_path": "$.article.headline_article",
            "order_case": """
                CASE json_extract_string(value, '$.article.headline_article')
                    WHEN 'Low' THEN 1
                    WHEN 'Medium' THEN 2
                    WHEN 'High' THEN 3
                    ELSE 99
                END
            """,
        },
    }

    for category, cfg in chart_categories.items():
        charts[category] = {}

        for group_by, key_expr in groupings.items():
            query = f"""
            SELECT
                json_extract_string(value, '{cfg["json_path"]}') AS metric,
                {key_expr} AS key,
                COUNT(*) AS count
            FROM results,
                 json_each(json_extract(result, '$.evaluations'))
            WHERE json_extract_string(value, '{cfg["json_path"]}') IS NOT NULL
            GROUP BY
                json_extract_string(value, '{cfg["json_path"]}'),
                {key_expr}
            ORDER BY
                {cfg["order_case"]},
                {key_expr};
            """

            rows = db.chart_query(query)
            charts[category][group_by] = db.transform_for_chart(rows)

    return charts


@app.get('/health')
async def health():
    return {"ok": True, "models": {"openai": bool(models.openai_client), "anthropic": bool(models.anthropic_client), "ollama": bool(models.ollama_client)}}
