# Research Intelligence Engine

A research assistant that turns a question into a detailed, source-linked literature review. The HTML interface is served by a Python API that runs the research pipeline and keeps AI credentials on the server.

## What it does

- IBM Granite on IBM Cloud watsonx.ai plans searches, extracts evidence, and drafts a report.
- Semantic Scholar provides academic paper metadata and abstracts; Crossref is used as a fallback.
- Each result includes source links so readers can review the original record.
- IBM Granite writes a detailed, evidence-grounded review with linked references.
- The responsive HTML interface is served by the FastAPI application.

The full workflow requires the Python API. GitHub Pages cannot run the research pipeline or protect WatsonX credentials. Keep credentials on the API server, never in the website or GitHub repository.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set these values in `.env`:

```dotenv
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_PROJECT_ID=your-project-uuid-v4
WATSONX_URL=https://your-region.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-4-h-small
OPENALEX_API_KEY=your-openalex-api-key
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-api-key
CROSSREF_MAILTO=your-contact-email
```

Use a project UUIDv4 from IBM Cloud, and associate the project with an active Watson Machine Learning service instance in the same region. Keep credentials in `.env`; never put real values in `.env.example` or commit them.

The main workflow uses Semantic Scholar with Crossref as its fallback. Configure a Semantic Scholar key and a Crossref contact email in the backend environment for more reliable public use. OpenAlex remains available to the optional Phase 2 workflow. Requests are paced and temporary rate limits or service errors are retried, but no software can guarantee unlimited availability from external research APIs.

## Run the HTML website and research API locally

```powershell
uvicorn app.api.main:app --reload --port 8000
```

Open <http://localhost:8000>. The API documentation is at <http://localhost:8000/docs>. The first research run may download the sentence-embedding model.

The earlier Streamlit interface remains available with `streamlit run app/streamlit_app.py`.

## Run with Docker

```powershell
docker compose up --build
```

Open <http://localhost:8000>.

## Deploy the full website and API

The repository includes a Render Blueprint in `render.yaml`. After this repository is connected to GitHub, create a Blueprint in Render from the repository. Enter `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` as private environment values when Render prompts; do not add them to GitHub. The full HTML website and API run together as one Python service. Its root page is served by FastAPI and uses the same service for research requests.

The Blueprint uses Render's free web service and a lightweight lexical ranker so the API does not need to load a sentence-embedding model. Free services sleep after 15 minutes without traffic, so the first request after idle can take longer. For always-on usage, choose a larger plan in Render and confirm its price before deploying.

## Optional later-phase components

Install optional components only when working on those parts of the project:

```powershell
pip install -e ".[advanced]"    # LangGraph workflow and evidence graph
pip install -e ".[api]"         # FastAPI service
pip install -e ".[evaluation]"  # RAGAS evaluation dependencies
pip install -e ".[dev]"         # pytest tooling
```

## Project structure

```text
index.html           Responsive HTML interface served by FastAPI
app/                 FastAPI backend and earlier Streamlit interface
rie/agents/          Planning, evidence extraction, and report generation
rie/llm/             IBM Granite / watsonx.ai client
rie/retrieval/       Academic sources, chunking, embeddings, and FAISS search
rie/graph/           Optional later-phase workflows
rie/evaluation/      Optional evaluation utilities
docs/                Architecture, API, and evaluation notes
tests/               Unit and integration tests
```

## Evaluation and limitations

Evaluation metrics are documented as a plan; no benchmark results are claimed. The MVP retrieves paper metadata and abstracts, not full papers. Review the linked source papers before relying on extracted claims, and treat results as an aid to research rather than a substitute for verification.
