# Business Analytics Copilot

Ask questions about tabular business data in plain English. The application profiles uploaded CSV or Excel files, loads them into DuckDB, and uses an LLM to plan and execute bounded, read-only SQL analysis. Answers are checked against query evidence, with result tables, charts, SQL, and an execution trace available in the Streamlit interface.

## Product walkthrough

1. Upload one or more CSV or XLSX files.
2. Review the detected tables, schemas, and sample rows.
3. Ask a question such as “Which region generated the highest revenue?”
4. Inspect the answer, supporting results, charts, generated SQL, and validation/execution trace.
5. Use **Clear data and reset session** when finished.

See the [demo guide](docs/demo.md) for a complete local walkthrough and example questions.

## Architecture

```text
CSV/XLSX → validation and profiling → session DuckDB
         → analysis plan → generated read-only SQL → validation → DuckDB
         → evidence checks → answer, result tables, and charts
```

The same analytics and database services are reused by the UI and Python entry points. See [docs/architecture.md](docs/architecture.md) for component responsibilities and deployment boundaries.

## Current capabilities

- CSV/XLSX ingestion, column normalization, validation, schema inspection, and DuckDB storage.
- OpenRouter-backed natural-language planning and DuckDB SQL generation.
- Read-only query validation against the active database schema before execution.
- Bounded multi-step analyses (four steps by default), compact follow-up context, and evidence-checked answers.
- Streamlit chat interface with supporting tables, predefined Plotly charts, SQL transparency, and trace details.
- A deterministic, no-network benchmark with 27 questions, including unsupported-question and multi-step cases.

Not implemented: authentication, persistent managed storage, multi-agent orchestration, RAG/vector search, or guaranteed causal inference. The sample data is synthetic. Root-cause analyses describe patterns and decompositions in the data; they do not prove causation.

## Requirements and local setup

Python 3.11 or later is recommended. From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set the OpenRouter values in `.env`:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

The API key is required for live LLM-backed questions. The model is configurable; `openrouter/free` is an example, not a quality or availability guarantee. Never commit `.env` or paste credentials into source files.

### Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, upload files, and ask a question. The application does not execute SQL directly from the browser: generated statements are validated by the backend before DuckDB execution. Uploaded session databases are stored under `data/processed/ui_sessions/` and can be removed with **Clear data and reset session**.

### Python entry points

With the configured sample database available:

```powershell
.\.venv\Scripts\python.exe -c "from src.analytics import answer_question; result = answer_question('Which region generated the highest revenue?'); print(result.data if result.success else result.error)"
```

For a bounded multi-step analysis:

```powershell
.\.venv\Scripts\python.exe -c "from src.analytics import AnalysisAgent; result = AnalysisAgent().answer_question('Why did profit decline in Q3?'); print(result.answer if result.success else result.error)"
```

The deterministic sample data is already supplied in `data/sample/`. To create or reset the default DuckDB database from those CSVs:

```powershell
.\.venv\Scripts\python.exe tests\test_data_foundation.py
```

## Configuration and secrets

Configuration precedence is process environment / local `.env`, then Streamlit secrets, then non-secret defaults where applicable. The OpenRouter settings are `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_BASE_URL`. In Streamlit Community Cloud, add them under the app's **Settings → Secrets** using TOML syntax; see the [deployment guide](docs/demo.md#deploy-to-streamlit-community-cloud). A blank, optional template is provided at `.streamlit/secrets.toml.example`; copy it to `.streamlit/secrets.toml` for local Streamlit secrets. The actual secrets file and `.env` are ignored by Git.

The API key is sent to OpenRouter for model requests. Uploaded business data is sent only as schema, compact query context, and bounded result summaries when needed; it is not sent as a whole dataset. Do not upload sensitive or regulated information to a hosted model without the required organizational approval.

## Tests and deterministic evaluation

Run the complete deterministic test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Run the no-network evaluation:

```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```

The latest recorded deterministic-fixture run reports 22/22 successful answerable SQL executions, 22/22 matching result sets, 22/22 evidence checks, 5/5 expected unsupported refusals, and 3/3 multi-step analyses. Average local latency in that run was 0.1275 seconds. These controlled fixtures verify pipeline behavior; they do **not** estimate hosted-model quality, live semantic accuracy, or production latency. Details and per-question output are in [evaluation/README.md](evaluation/README.md) and `evaluation/results/`.

## Streamlit Community Cloud

1. Push this project to a GitHub repository using your normal workflow.
2. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the main file.
3. In the app settings, add the three OpenRouter settings in TOML form. Do not place credentials in repository files.
4. Deploy, upload a small CSV/XLSX dataset, and test one supported and one unsupported question.

Community Cloud's local filesystem is ephemeral and is not a managed database. Uploaded session DuckDB files may disappear when an instance restarts and should not be treated as durable storage. This project has no authentication or multi-tenant access controls; use only appropriate non-sensitive demo data. See the [demo and deployment guide](docs/demo.md) for more detail.

## Project status

Phases 1–5 (data foundation, NL-to-SQL, bounded analysis, Streamlit UI, and deterministic evaluation) are implemented. Phase 6 prepares the documentation and deployment configuration; production hosting, authentication, durable storage, and live-model quality guarantees are outside the current scope.
