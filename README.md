🚀 AI Data Assistant – Natural Language Analytics Platform
🔍 Overview

AI Data Assistant is an enterprise-grade analytics application that allows business users to query databases using plain English, automatically generating SQL, visualizations, and decision-ready insights — without writing a single line of SQL.

This project focuses on intent understanding, self-healing SQL generation, smart visualization selection, and privacy-first AI deployment using a local LLM.

🧠 Core Architecture

![Architecture Diagram](assets/architecture.png)

⚙️ Key Features
🔹 Natural Language → SQL

Converts business questions into optimized MySQL queries

Auto-enriches results with contextual metrics

Prevents unsafe SQL (DROP, DELETE, UPDATE, etc.)

🔹 Self-Healing Query Engine

Automatically fixes SQL errors using LLM feedback

Retries execution without user intervention

🔹 Intelligent Visualization Engine

Auto-selects chart type (Bar, Line, Scatter, Donut, Area, Combo)

Supports multi-metric alignment & aggregation

Smart legends, color themes, and scaling

Fully interactive Plotly dashboards

🔹 Strategic Decision Layer

Generates:

Observation (what is happening)

Insight (why it matters)

Recommendation (what to do next)

🔹 Enterprise-Grade UI

Dark theme dashboard

KPI cards

Export to CSV / JSON

SQL audit visibility

🔹 Privacy-First AI

Uses local LLM (Qwen-2.5 via Ollama)

No data leaves the machine

🛠️ Tech Stack

Frontend: Streamlit, Plotly

Backend: Python, SQLAlchemy

Database: MySQL

AI Model: Qwen-2.5 (Local via Ollama)

Analytics: Pandas, NumPy

📸 Screenshots

### Natural Language Query to  AI-Generated Insights & Visualization
![NL to AI-Generated Insights & Visualization](assets/screenshots/'FullUI-1.png')

![NL to AI-Generated Insights & Visualization](assets/screenshots/'FullUI-2.png')

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py


Ensure:

MySQL is running

Ollama is running locally

Qwen-2.5 model is available

🎯 Use Cases

Sales & Revenue Analysis

HR Attrition Insights

Finance Performance Tracking

Operations & KPI Monitoring

👤 Author

Santhosh C
Data Analyst | AI Automation | SQL Intelligence
