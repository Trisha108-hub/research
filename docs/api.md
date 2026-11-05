# API Documentation

## Run the API

```powershell
pip install fastapi uvicorn
uvicorn app.api.main:app --reload --port 8000
```

The HTML website is available at <http://localhost:8000>, and interactive API documentation is at <http://localhost:8000/docs>.

## Endpoints

### `POST /research/run`

Run the Phase 1 pipeline, or set `phase2` to `true` for the full LangGraph pipeline.

```json
{
  "question": "How does retrieval-augmented generation affect factual accuracy?",
  "results_per_source": 10,
  "top_k": 5,
  "phase2": false
}
```

### `POST /research/jobs` and `GET /research/jobs/{job_id}`

The HTML website uses the job endpoints to receive staged progress updates while a literature review is running. Start a job with the same JSON body as `/research/run`, then poll its job ID until the response status is `complete` or `failed`.

### `POST /reports/evaluate`

Evaluate generated text with citation accuracy and RAGAS metrics.

### `GET /research/health`

Returns the API health status.

## Docker

```powershell
docker compose up --build
```

- API: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
