# Demo and deployment guide

## Run locally

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Configure the OpenRouter connection in `.env` (copy `.env.example` only if `.env` does not already exist):

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in local values. Never commit either secrets file. Environment / `.env` values take precedence over Streamlit secrets.

Launch the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Suggested walkthrough

Use the supplied synthetic files from `data/sample/`:

1. In the sidebar, upload `orders.csv`, `customers.csv`, `products.csv`, and `targets.csv`, then select **Load data**.
2. Confirm the detected tables and columns in the sidebar; review the sample previews.
3. Ask **“Which region generated the highest revenue?”**
4. Review the answer and evidence check, then expand **Supporting query results**, **Analysis trace**, and **SQL queries used**.
5. Try **“What were the monthly revenues?”**, **“Which regions missed their revenue target?”**, or **“Which customer segment generates the most revenue?”**
6. Try **“Which employees had the highest salary?”** to see a refusal when the uploaded schema does not support the question.
7. Select **Clear data and reset session** to remove the session database and clear the conversation.

The hosted model can vary in response quality and availability. If a request fails, check the configured model and provider availability; deterministic tests do not require an API key.

## Verify locally

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```

The evaluation is deterministic and makes no OpenRouter requests. See [evaluation/README.md](../evaluation/README.md) for fixture scope and interpretation.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository using your normal release process. Keep `.env` and `.streamlit/secrets.toml` out of the repository.
2. In Streamlit Community Cloud, create an app and select the repository, deployment branch, and `app.py` as the main file.
3. Open the app's **Settings → Secrets** and enter:

   ```toml
   OPENROUTER_API_KEY = "your-key"
   OPENROUTER_MODEL = "openrouter/free"
   OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
   ```

4. Save the settings and deploy/reboot the app.
5. Upload the four small sample CSV files and run one supported question and one unsupported question. Confirm the answer evidence status and inspect the SQL/trace panels.

The repository requirements install the runtime packages during deployment. The sample CSVs are included in the repository and do not require an API call to load. Keep the OpenRouter key only in the Cloud secrets settings and rotate it through the provider if it is ever exposed.

### Cloud limitations

- The uploaded DuckDB files live on the app's writable local filesystem. Community Cloud storage is ephemeral, so files can be lost on restart/redeployment and must not be treated as durable.
- Uploaded data and question context should be treated as accessible to the hosted app and its configured model provider. Use only non-sensitive demo data unless the relevant data-handling review explicitly approves otherwise.
- The app has no authentication, role-based access, durable database, or production multi-tenant controls.
- Hosted free-model availability, rate limits, SQL quality, and response latency are not guaranteed.
- A successful deterministic evaluation is not a live-model or production reliability guarantee.
