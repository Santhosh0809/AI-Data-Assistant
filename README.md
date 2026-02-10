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

### 1️⃣ Prerequisites
Make sure the following are installed on your system:

- Python **3.9 or above**
- Git
- Internet connection (required only for first-time model download)

Verify Python:
```bash
python --version
```

---

### 2️⃣ Install Ollama (Local LLM Runtime)

Download and install Ollama from the official site:
https://ollama.com

Verify installation:
```bash
ollama --version
```

---

### 3️⃣ Download the AI Model (Qwen-2.5)

Pull the required local LLM model:
```bash
ollama pull qwen2.5-coder:7b
```

Confirm the model is available:
```bash
ollama list
```

---

### 4️⃣ Start Ollama Server

Start the Ollama service:
```bash
ollama serve
```

⚠️ **Important:**  
Keep this terminal **running**. Do not close it while using the application.

---

### 5️⃣ Install Project Dependencies

Navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

---

### 6️⃣ Run the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Open the application in your browser:
```text
http://localhost:8501
```

You can now ask business questions based on your database information in plain English and get SQL, visualizations, and insights.
## 🧪 Example Queries
- Show monthly sales trends by region
- Which products have declining revenue?
- Compare revenue and profit by category
- Top 5 customers by total spend

## 👤 Author

**Santhosh C**  
Data Analyst | AI Automation | SQL Intelligence
