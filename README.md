# AI Data Analysis Platform
![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Groq](https://img.shields.io/badge/Groq-LLM-black)

An end-to-end data analysis workspace built with Streamlit — upload a CSV or Excel file and move from raw data to cleaned datasets, visualizations, quality checks, ML predictions, and AI-powered summaries.

## Features

| Module | What it does |
| --- | --- |
| Data Cleaning | Handle duplicates, missing values, and data types with auto or manual control |
| Outlier Detection | Detect and review anomalies using IQR-based logic with reversible actions |
| Data Quality | Surface completeness, uniqueness, and consistency signals |
| Visualization | Generate interactive Plotly charts with smart column-type detection |
| AI Insights | Ask natural language questions about your dataset powered by Groq LLM |
| Predictions | Train and evaluate regression models with feature-level interpretation |
| Export Report | Download a full analysis report with charts and summaries |

## Setup

```bash
git clone <your-repo-url>
cd <project-folder>
pip install -r requirements.txt
```

Then create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at https://console.groq.com

Then run:

```bash
streamlit run app.py
```

## Deploying to Streamlit Cloud

Instead of a `.env` file, paste the key into the Secrets panel in your Streamlit Cloud dashboard under `GROQ_API_KEY = "your_key_here"`. The app reads it automatically.

## Deployment Checklist

1. Push this project to a GitHub repository.
2. Make sure these files are present in the repo root:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
3. Do not push real secrets. Keep `.env` and `.streamlit/secrets.toml` out of Git.
4. In Streamlit Cloud, create a new app and select:
   - Repository: your GitHub repo
   - Branch: your deploy branch
   - Main file path: `app.py`
5. In the app dashboard, open `Settings -> Secrets` and add:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

6. Click `Deploy`.

## Notes For Successful Cloud Deployment

- Python is pinned through `runtime.txt` to improve consistency with local development.
- Uploaded datasets work normally in Streamlit Cloud, but files saved only in session state are temporary and reset when the app restarts.
- If deployment fails during install, redeploy after checking package versions in `requirements.txt`.
- If Groq features do not work after deployment, verify the secret name is exactly `GROQ_API_KEY`.

## Project structure

```text
.
├── app.py
├── pages/
│   ├── 1_Data_Cleaning.py
│   ├── 2_Outlier_Detection.py
│   ├── 3_Data_Quality.py
│   ├── 4_Visualization.py
│   ├── 5_AI_Insights.py
│   ├── 6_Predictions.py
│   └── 7_Export_Report.py
├── utils/
│   ├── chart_summary.py
│   ├── column_classifier.py
│   ├── heatmap.py
│   ├── helpers.py
│   ├── logger.py
│   ├── styles.py
│   └── visualization.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── .env
└── requirements.txt
```

## Credits

Built by: Shubham Tiwari, Shrajal Raghuwanshi, Parth Thakur, Prathamesh Rathod

Shri Ramdeobaba College of Engineering and Management — Department of Data Science, 2025–26

Guide: Prof. Shruti Kolte
