# 🚀 AI Data Assistant  
### Natural Language Analytics Platform

---

## 🔍 Overview

**AI Data Assistant** is an enterprise-grade analytics application that allows business users to query databases using **plain English**.  
It automatically generates **SQL queries, visualizations, and decision-ready insights** — without writing a single line of SQL.

This project focuses on:
- Intent understanding  
- Self-healing SQL generation  
- Intelligent visualization selection  
- Privacy-first AI using a **local LLM**

---

## 🧠 Core Architecture

![Architecture Diagram](assets/architecture.png)

---

## ⚙️ Key Features

### 🔹 Natural Language → SQL
- Converts business questions into optimized **MySQL queries**
- Auto-enriches results with contextual metrics
- Prevents unsafe SQL operations (`DROP`, `DELETE`, `UPDATE`, etc.)

---

### 🔹 Self-Healing Query Engine
- Automatically fixes SQL errors using **LLM feedback**
- Retries execution without user intervention

---

### 🔹 Intelligent Visualization Engine
- Auto-selects best chart type:
  - Bar, Line, Scatter, Donut, Area, Combo
- Supports **multi-metric aggregation & alignment**
- Smart legends, scaling, and color themes
- Fully interactive **Plotly dashboards**

---

### 🔹 Strategic Decision Layer
Automatically generates:
- **Observation** – What is happening  
- **Insight** – Why it matters  
- **Recommendation** – What to do next  

---

### 🔹 Enterprise-Grade UI
- Dark-themed dashboard
- KPI cards
- Export results to **CSV / JSON**
- SQL audit visibility

---

### 🔹 Privacy-First AI
- Uses **local LLM (Qwen-2.5 via Ollama)**
- No data leaves the machine
- Fully offline and secure

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly  
- **Backend:** Python, SQLAlchemy  
- **Database:** MySQL  
- **AI Model:** Qwen-2.5 (Local via Ollama)  
- **Analytics:** Pandas, NumPy  

---

## 📸 Application Screenshots

### Natural Language Query → AI Insights & Visualization

![Full UI 1](assets/screenshots/FullUI-1.png)  
![Full UI 2](assets/screenshots/FullUI-2.png)

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
