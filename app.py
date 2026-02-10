import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ollama
import json
import re
from sqlalchemy import create_engine, text
from datetime import date, datetime
import numpy as np

# =========================================================
# 1. CORE CONFIG - ENTERPRISE THEME
# =========================================================
st.set_page_config(
    page_title="AI Data Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Color Palette & CSS
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    
    .kpi-card {
        background: linear-gradient(135deg, #1E2130 0%, #2D3142 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00D4AA;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    
    .insight-card {
        background: #1A1D29;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2D3142;
        margin-bottom: 12px;
    }
    .insight-header {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8B92A8;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .insight-text { color: #E4E6F0; font-size: 14px; line-height: 1.5; }
    
    .viz-container {
        background: #161922;
        border-radius: 12px;
        border: 1px solid #2D3142;
        padding: 20px;
        margin-top: 20px;
    }
    
    .dataframe { 
        background-color: #1A1D29 !important; 
        border: 1px solid #2D3142 !important;
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0F1117; }
    ::-webkit-scrollbar-thumb { background: #2D3142; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3D4256; }
    
    /* Toggle Switch Styling */
    .stToggle > label {
        font-size: 16px !important;
        color: #E4E6F0 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DATABASE CONNECTION
# =========================================================
@st.cache_resource
def get_engine():
    return create_engine(
        "mysql+pymysql://user:password@host/db_name", #Use Your Database credentials
        pool_recycle=3600,
        pool_pre_ping=True
    )

engine = get_engine()

if "state" not in st.session_state:
    st.session_state.state = {"plan": None, "df": None, "viz_config": {}, "show_viz": False}

# =========================================================
# 3. COLOR UTILITY FUNCTIONS
# =========================================================
def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color to rgba tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def get_color_sequence(theme_name):
    """Return color sequence based on theme"""
    themes = {
        "Enterprise Dark": ["#00D4AA", "#0078D4", "#F8B100", "#107C10", "#8B92A8", "#FF6B6B", "#4ECDC4"],
        "Ocean Blue": ["#0066CC", "#0080FF", "#3399FF", "#66B2FF", "#99CCFF", "#004C99", "#003366"],
        "Forest Green": ["#107C10", "#14A014", "#2DB92D", "#47D147", "#61E861", "#0A5A0A", "#063806"],
        "Sunset": ["#FF6B35", "#F7931E", "#FFD23F", "#EE4266", "#540D6E", "#F7B801", "#F18701"],
        "Monochrome": ["#00D4AA", "#0078D4", "#F8B100", "#107C10", "#8B92A8", "#FF6B6B", "#4ECDC4"]
    }
    return themes.get(theme_name, themes["Enterprise Dark"])

# =========================================================
# 4. PRECISION ANALYTICS PROTOCOL
# =========================================================
@st.cache_data(show_spinner=False)
def get_analysis(user_query, _schema_text, _sample_data, retry_error=None):
    feedback = f"\n\nPREVIOUS ERROR TO FIX:\n{retry_error}" if retry_error else ""

    prompt = f"""
You are a UNIVERSAL DATA ARCHITECT. Transform any natural language question into valid SQL for ANY database schema.

========================
AUTHORITATIVE SCHEMA (GROUND TRUTH)
========================
{_schema_text}

SAMPLE DATA (UNDERSTAND VALUE RANGES, FORMATS, NULLS):
{_sample_data}

========================
USER QUESTION
========================
{user_query}

========================
INTENT ANALYSIS & COLUMN ENRICHMENT
========================

STEP 1: DECOMPOSE INTENT
Identify the core need:
- PRIMARY METRIC: What main number does user want? (revenue, count, average, ratio)
- PRIMARY DIMENSION: How to slice it? (time, category, region, product)
- CONTEXT NEEDED: What additional info helps decision-making?

STEP 2: AUTO-ENRICHMENT RULE
Always include 3-5 complementary columns that provide context:

If user asks for SALES/REVENUE, also include:
- Quantity/Volume (understand scale)
- Profit/Margin (understand efficiency)
- Discount (understand pricing strategy)
- Order Count (understand transaction frequency)

If user asks for GROWTH/TREND, also include:
- Previous period value (baseline comparison)
- Absolute change (magnitude)
- Percentage change (rate)
- Contributing factors (what drove change)

If user asks for PERFORMANCE/RANKING, also include:
- Previous rank (trend in position)
- Gap to next/prev (competitive context)
- Percentile (relative standing)
- Trend direction (improving/declining)

If user asks for COMPARISON, also include:
- Variance amount (difference size)
- Variance percent (relative difference)
- Contribution to total (share of whole)
- Historical average (context)

UNIVERSAL CONTEXT COLUMNS (always add when available):
- Record count (sample size)
- Date range (time context)
- Category breakdown (segmentation)

========================
CONCEPTUAL FRAMEWORK (APPLY TO ANY DOMAIN)
========================

[1. INTENT DECOMPOSITION]
Break the question into abstract components:
- MEASURE: What numeric property to quantify? (count, sum, average, ratio of what?)
- DIMENSION: How to categorize/slice the data? (by time, by category, by entity?)
- FILTER: Which records to include/exclude? (time range, threshold conditions?)
- CALCULATION: Any derived values needed? (percentages, differences, rates?)

Examples across domains:
- Retail: "total sales by region" → MEASURE=sum of transaction values, DIMENSION=geographic area
- Healthcare: "average patient stay by department" → MEASURE=mean duration, DIMENSION=organizational unit  
- Finance: "monthly portfolio growth" → MEASURE=change in value, DIMENSION=time period, CALCULATION=growth rate
- Manufacturing: "defect rate by production line" → MEASURE=ratio of bad to total, DIMENSION=production unit

[2. SCHEMA MAPPING]
Map abstract concepts to actual schema:
- Find columns representing the MEASURE (usually numeric types)
- Find columns representing the DIMENSION (usually categorical/date types)
- Find columns usable for FILTER (date, status, value thresholds)
- Find columns for AUTO-ENRICHMENT (context providers per rules above)

ABSOLUTE RULES:
- Column names in schema are EXACT and CASE-SENSITIVE
- Use backticks: `Column Name` (preserves spaces)
- Never "clean" or rename columns

[3. SQL CONSTRUCTION PATTERNS]

PATTERN A - Simple Aggregation with Enrichment:
SELECT 
    `Dimension_Column`,
    SUM(`Primary_Value`) as Total_Primary,
    SUM(`Secondary_Value`) as Supporting_Metric_1,
    AVG(`Context_Value`) as Supporting_Metric_2,
    COUNT(*) as Record_Count,
    (SUM(`Primary_Value`) / NULLIF(SUM(`Secondary_Value`), 0)) as Efficiency_Ratio
FROM `Table`
GROUP BY `Dimension_Column`
ORDER BY Total_Primary DESC
LIMIT 1000

PATTERN B - Time Series with Context:
SELECT 
    DATE_FORMAT(`Date_Column`, '%Y-%m') as Time_Period,
    SUM(`Current_Value`) as Period_Total,
    LAG(SUM(`Current_Value`)) OVER (ORDER BY DATE_FORMAT(`Date_Column`, '%Y-%m')) as Previous_Period,
    SUM(`Current_Value`) - LAG(SUM(`Current_Value`)) OVER (ORDER BY DATE_FORMAT(`Date_Column`, '%Y-%m')) as Absolute_Change,
    ((SUM(`Current_Value`) - LAG(SUM(`Current_Value`)) OVER (ORDER BY DATE_FORMAT(`Date_Column`, '%Y-%m'))) / 
     NULLIF(LAG(SUM(`Current_Value`)) OVER (ORDER BY DATE_FORMAT(`Date_Column`, '%Y-%m')), 0)) * 100 as Growth_Percent,
    COUNT(DISTINCT `Entity_ID`) as Unique_Entities,
    AVG(`Related_Metric`) as Avg_Context
FROM `Table`
WHERE `Date_Column` >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY Time_Period
ORDER BY Time_Period
LIMIT 1000

PATTERN C - Multi-Dimensional Analysis:
SELECT 
    `Primary_Dimension`,
    `Secondary_Dimension`,
    SUM(`Main_Metric`) as Primary_Total,
    SUM(`Supporting_Metric_1`) as Context_1,
    AVG(`Supporting_Metric_2`) as Context_2,
    COUNT(*) as Volume,
    SUM(`Main_Metric`) / NULLIF(SUM(SUM(`Main_Metric`)) OVER (), 0) * 100 as Share_of_Total,
    RANK() OVER (ORDER BY SUM(`Main_Metric`) DESC) as Rank_Position
FROM `Table`
GROUP BY `Primary_Dimension`, `Secondary_Dimension`
ORDER BY Primary_Total DESC
LIMIT 1000

[4. DERIVED METRIC PRINCIPLES]
When user asks for something not directly in schema:

STEP 1: Can it be calculated from existing columns?
- Total = Unit × Quantity
- Margin = Revenue − Cost
- Rate = Part / Whole × 100
- Growth = (Current − Previous) / Previous × 100

STEP 2: Can it be inferred from data patterns?
- High/Medium/Low from percentile thresholds
- Active/Inactive from date recency
- Repeated/Single from count of occurrences

STEP 3: If neither possible
- Remove the constraint
- Explain in decision_support.recommendation what data would be needed

[5. SAFETY CONSTRAINTS]
- ALWAYS use LIMIT 1000 (unless user explicitly says "all data")
- NEVER use: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT
- Use NULLIF(denominator, 0) for all divisions
- Use COALESCE(column, 0) for NULL handling in aggregates

[6. ALIAS NAMING]
- Simple and descriptive: Total_Value not Calculated_Total_Value_This_Period
- No special characters in alias names
- No spaces in alias names (use underscore)

========================
VISUALIZATION SELECTION (UNIVERSAL)
========================
Based on data structure, not content:

- One dimension + one measure + time element → Line chart
- One dimension + one measure (no time) → Bar chart
- Part-to-whole relationship → Pie/Donut chart
- Two measures comparison → Scatter plot
- One dimension + multiple measures → Grouped bar chart
- Time dimension + multiple categories → Stacked area chart

========================
OUTPUT FORMAT (STRICT JSON)
========================
{{
  "title": "Concise description of what this analysis shows",
  "sql": "Valid MySQL SELECT statement ending with LIMIT 1000",
  "enrichment_applied": "Description of extra columns added for context",
  "decision_support": {{
      "observation": "Pattern visible in data (what is happening)",
      "insight": "Interpretation of pattern (why it matters)",
      "recommendation": "Suggested action based on insight"
  }},
  "viz": {{
      "type": "Bar|Line|Scatter|Pie|Area|Combo|Table",
      "x": "dimension_column_name",
      "y": ["primary_measure", "context_measure_1", "context_measure_2"],
      "color": "optional_category_column_for_grouping"
  }}
}}

{feedback}

========================
EXECUTION INSTRUCTION
========================
Analyze the schema, identify user intent, apply auto-enrichment rules, generate SQL with complementary columns, output valid JSON.
"""


    res = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0}
    )

    try:
        clean = re.sub(r'^```json|```$', '', res['message']['content'], flags=re.MULTILINE)
        return json.loads(re.search(r'\{.*\}', clean, re.DOTALL).group(0))
    except Exception as e:
        if not retry_error:
            return get_analysis(user_query, _schema_text, _sample_data, retry_error=str(e))
        return {"error": str(e)}

# =========================================================
# 5. METADATA EXTRACTION
# =========================================================
@st.cache_data(ttl=600)
def get_cached_metadata():
    with engine.connect() as conn:
        tables = pd.read_sql("SHOW TABLES", conn).iloc[:, 0].tolist()
        schema_parts, details, samples = [], {}, {}

        for t in tables:
            cols = pd.read_sql(f"SHOW COLUMNS FROM `{t}`", conn)
            schema_parts.append(f"Table `{t}` Columns: {cols['Field'].tolist()}")
            details[t] = cols[['Field', 'Type', 'Key']]

            df_sample = pd.read_sql(f"SELECT * FROM `{t}` LIMIT 3", conn)
            df_sample = df_sample.applymap(
                lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, date)) else x
            )
            samples[t] = df_sample.to_dict()

        return "\n".join(schema_parts), details, str(samples)

schema_text, table_details, sample_data = get_cached_metadata()

# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; border-bottom: 1px solid #2D3142; margin-bottom: 20px;'>
        <h2 style='color: #00D4AA; margin: 0; font-size: 24px;'>🏛️ ADA</h2>
        <p style='color: #8B92A8; font-size: 12px; margin: 5px 0 0 0;'>AI-POWERED ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Data Models")
    for t, d in table_details.items():
        with st.expander(f"**{t}**", expanded=False):
            st.dataframe(d, hide_index=True, use_container_width=True)

# =========================================================
# 7. MAIN INTERFACE
# =========================================================
st.markdown("""
<div style='margin-bottom: 10px;'>
    <h1 style='color: #E4E6F0; margin: 0; font-weight: 700;'>AI Data Assistant</h1>
    <p style='color: #8B92A8; font-size: 14px; margin: 5px 0 0 0;'>Natural language to actionable insights</p>
</div>
""", unsafe_allow_html=True)

col_input, col_btn = st.columns([4, 1])
with col_input:
    user_input = st.text_input(
        "", 
        placeholder="Ask a business question (e.g., 'Show Q4 revenue trends by region')...",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_btn = st.button("🚀 Generate Intelligence", use_container_width=True, type="primary")

if analyze_btn and user_input:
    with st.spinner("Analyzing business context..."):
        plan = get_analysis(user_input, schema_text, sample_data)

        if "error" not in plan:
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(text(plan["sql"]), conn)
                st.session_state.state = {"plan": plan, "df": df, "viz_config": {}, "show_viz": False}
                st.success("Analysis complete")
            except Exception as e:
                st.error(f"Execution failed: {e}")
                st.session_state.state = {"plan": plan, "df": pd.DataFrame(), "viz_config": {}, "show_viz": False}
        else:
            st.error(f"Analysis error: {plan['error']}")

# =========================================================
# 8. OUTPUT SECTION
# =========================================================
if st.session_state.state["df"] is not None:
    plan, df = st.session_state.state["plan"], st.session_state.state["df"]
    ds, viz = plan.get("decision_support", {}), plan.get("viz", {})

    if not df.empty:
        # --- KPI SECTION ---
        st.markdown("### 📊 Key Performance Indicators")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        kpi_cols = st.columns(min(4, len(numeric_cols) + 1))
        
        with kpi_cols[0]:
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='color: #8B92A8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>Total Records</div>
                <div style='color: #00D4AA; font-size: 28px; font-weight: 700; margin-top: 5px;'>{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        for idx, col in enumerate(numeric_cols[:3], 1):
            if idx < len(kpi_cols):
                total = df[col].sum()
                avg = df[col].mean()
                with kpi_cols[idx]:
                    st.markdown(f"""
                    <div class='kpi-card' style='border-left-color: #0078D4;'>
                        <div style='color: #8B92A8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>{col}</div>
                        <div style='color: #E4E6F0; font-size: 24px; font-weight: 700; margin-top: 5px;'>{total:,.0f}</div>
                        <div style='color: #0078D4; font-size: 12px; margin-top: 5px;'>Avg: {avg:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # --- DECISION SUPPORT INSIGHTS ---
        st.markdown("### 💡 Strategic Insights")
        c1, c2, c3 = st.columns(3)
        
        insights = [
            ("🔍 Observation", ds.get('observation', 'No observation available'), '#00D4AA'),
            ("💡 Insight", ds.get('insight', 'No insight available'), '#F8B100'), 
            ("🎯 Recommendation", ds.get('recommendation', 'No recommendation available'), '#107C10')
        ]
        
        for col, (title, content, color) in zip([c1, c2, c3], insights):
            col.markdown(f"""
            <div class='insight-card'>
                <div class='insight-header' style='color: {color};'>{title}</div>
                <div class='insight-text'>{content}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- DATA TABLE SECTION ---
        st.markdown(f"### 📋 {plan.get('title', 'Analysis Results')}")
        
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    help=f"Sum: {df[col].sum():,.0f}" if df[col].dtype in ['int64', 'float64'] else None,
                    format="%.2f" if df[col].dtype == 'float64' else "%d"
                ) for col in df.select_dtypes(include=[np.number]).columns
            }
        )
        
        # Export options
        col_dl1, col_dl2, _ = st.columns([1, 1, 4])
        with col_dl1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ CSV Export", csv, "analysis_data.csv", "text/csv", use_container_width=True)
        with col_dl2:
            json_data = df.to_json(orient='records')
            st.download_button("⬇️ JSON Export", json_data, "analysis_data.json", "application/json", use_container_width=True)

        # SQL Expander
        with st.expander("⚙️ Technical Audit (SQL)"):
            st.code(plan["sql"], language="sql")

        # --- VISUALIZATION SECTION WITH TOGGLE ---
        st.markdown("---")
        
        # Toggle switch for visualization
        show_viz = st.toggle(
            "📈 Launch Visualization", 
            value=st.session_state.state.get("show_viz", False),
            key="viz_toggle"
        )
        
        # Update session state
        st.session_state.state["show_viz"] = show_viz
        
        if show_viz:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            if not num_cols:
                st.warning("No numeric columns available for visualization.")
            else:
                # Layout: Controls (25%) | Visualization (75%)
                col_ctrl, col_viz = st.columns([1, 3])
                
                # --- CONTROL PANEL ---
                with col_ctrl:
                    st.markdown("### 🎛️ Visualization Controls")
                    
                    # Chart Type Selection
                    chart_types = {
                        "Auto (Recommended)": "auto",
                        "📊 Bar Chart": "bar", 
                        "📈 Line Chart": "line",
                        "🥧 Pie/Donut": "pie",
                        "📉 Area Chart": "area",
                        "🔘 Scatter Plot": "scatter",
                        "⚡ Combo (Bar + Line)": "combo"
                    }
                    
                    selected_chart = st.selectbox(
                        "Chart Type",
                        options=list(chart_types.keys()),
                        index=0
                    )
                    chart_type = chart_types[selected_chart]
                    
                    # Smart Defaults from LLM
                    llm_x = viz.get("x")
                    llm_y = viz.get("y", [])
                    llm_color = viz.get("color")
                    
                    # Dimension Selection
                    st.markdown("#### Dimensions & Metrics")
                    
                    x_default = llm_x if llm_x in all_cols else (cat_cols[0] if cat_cols else all_cols[0])
                    x_axis = st.selectbox(
                        "X Axis (Category)",
                        all_cols,
                        index=all_cols.index(x_default) if x_default in all_cols else 0
                    )
                    
                    # Y Axis
                    y_default = [c for c in llm_y if c in num_cols]
                    if not y_default and num_cols:
                        y_default = [num_cols[0]]
                    
                    y_axes = st.multiselect(
                        "Y Axis (Metrics)",
                        num_cols,
                        default=y_default
                    )
                    
                    # Aggregation
                    agg_method = st.selectbox(
                        "Aggregation",
                        ["Sum", "Average", "Count", "Min", "Max"],
                        index=0
                    )

                    # Map UI labels to pandas function names
                    pandas_agg_map = {
                        "Sum": "sum",
                        "Average": "mean",  # ← This is the key fix
                        "Count": "count",
                        "Min": "min",
                        "Max": "max"
                    }
                    
                    # Color/Grouping
                    color_options = [None] + cat_cols
                    color_default = llm_color if llm_color in cat_cols else None
                    color_axis = st.selectbox(
                        "Color / Group By",
                        color_options,
                        index=color_options.index(color_default) if color_default else 0
                    )
                    
                    # Styling Options
                    st.markdown("#### Styling Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_data_labels = st.toggle("Data Labels", value=True)
                    with col2:
                        show_grid = st.toggle("Grid Lines", value=True)
                    
                    if chart_type in ["bar", "combo"]:
                        barmode_option = st.radio(
                            "Bar Mode", 
                            ["Grouped", "Stacked"],
                            horizontal=True
                        )
                        # Map UI labels to Plotly API values
                        barmode_map = {"Grouped": "group", "Stacked": "stack"}
                        barmode = barmode_map[barmode_option]
                    else:
                        barmode = "group"
                    
                    # Color Theme
                    theme = st.selectbox(
                        "Color Theme",
                        ["Enterprise Dark", "Ocean Blue", "Forest Green", "Sunset", "Monochrome"],
                        index=0
                    )
                    
                    color_seq = get_color_sequence(theme)

                # --- VISUALIZATION ENGINE ---
                with col_viz:
                    if not y_axes:
                        st.info("👈 Please select at least one metric to generate visualization.")
                    else:
                        # Prepare data
                        pdf = df.copy()
                        
                        # Apply aggregation if color axis selected
                        if color_axis and agg_method != "None":
                            agg_dict = {y: pandas_agg_map[agg_method] for y in y_axes}
                            if color_axis != x_axis:
                                pdf = pdf.groupby([x_axis, color_axis])[y_axes].agg(agg_dict).reset_index()
                            else:
                                pdf = pdf.groupby([x_axis])[y_axes].agg(agg_dict).reset_index()
                        
                        try:
                            fig = go.Figure()
                            
                            # Auto-detect chart type
                            if chart_type == "auto":
                                if len(y_axes) == 1 and len(pdf) > 20:
                                    chart_type = "line"
                                elif color_axis and len(pdf) < 10:
                                    chart_type = "pie"
                                else:
                                    chart_type = "bar"
                            
                            # BAR CHART
                            if chart_type == "bar":
                                if color_axis:
                                    for i, cat in enumerate(pdf[color_axis].unique()):
                                        cat_df = pdf[pdf[color_axis] == cat]
                                        for y in y_axes:
                                            fig.add_trace(go.Bar(
                                                x=cat_df[x_axis],
                                                y=cat_df[y],
                                                name=f"{y} ({cat})",
                                                text=cat_df[y].round(2) if show_data_labels else None,
                                                textposition="outside",
                                                marker_color=color_seq[i % len(color_seq)],
                                                textfont=dict(size=10, color="#E4E6F0")
                                            ))
                                else:
                                    for i, y in enumerate(y_axes):
                                        fig.add_trace(go.Bar(
                                            x=pdf[x_axis],
                                            y=pdf[y],
                                            name=y,
                                            text=pdf[y].round(2) if show_data_labels else None,
                                            textposition="outside",
                                            marker_color=color_seq[i % len(color_seq)],
                                            textfont=dict(size=10, color="#E4E6F0")
                                        ))
                                
                                fig.update_layout(barmode=barmode)
                            
                            # LINE CHART (FIXED - No color parsing error)
                            elif chart_type == "line":
                                if color_axis:
                                    for i, cat in enumerate(pdf[color_axis].unique()):
                                        cat_df = pdf[pdf[color_axis] == cat].sort_values(x_axis)
                                        for y in y_axes:
                                            fig.add_trace(go.Scatter(
                                                x=cat_df[x_axis],
                                                y=cat_df[y],
                                                mode="lines+markers",
                                                name=f"{y} ({cat})",
                                                line=dict(width=3, color=color_seq[i % len(color_seq)]),
                                                marker=dict(size=8, line=dict(width=2, color="#0F1117")),
                                                hovertemplate='<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
                                            ))
                                else:
                                    for i, y in enumerate(y_axes):
                                        # FIXED: Use hex color directly without parsing
                                        line_color = color_seq[i % len(color_seq)]
                                        fig.add_trace(go.Scatter(
                                            x=pdf[x_axis],
                                            y=pdf[y],
                                            mode="lines+markers",
                                            name=y,
                                            line=dict(width=3, color=line_color),
                                            marker=dict(size=8, line=dict(width=2, color="#0F1117")),
                                            fill='tozeroy' if len(y_axes) == 1 else None,
                                            fillcolor=hex_to_rgba(line_color, 0.2) if len(y_axes) == 1 else None
                                        ))
                            
                            # AREA CHART
                            elif chart_type == "area":
                                for i, y in enumerate(y_axes):
                                    area_color = color_seq[i % len(color_seq)]
                                    fig.add_trace(go.Scatter(
                                        x=pdf[x_axis],
                                        y=pdf[y],
                                        fill='tozeroy',
                                        name=y,
                                        line=dict(width=2, color=area_color),
                                        fillcolor=hex_to_rgba(area_color, 0.3)
                                    ))
                            
                            # SCATTER PLOT
                            elif chart_type == "scatter":
                                if len(y_axes) > 0:
                                    scatter_color = color_seq[0]
                                    fig.add_trace(go.Scatter(
                                        x=pdf[x_axis],
                                        y=pdf[y_axes[0]],
                                        mode='markers',
                                        name=y_axes[0],
                                        marker=dict(
                                            size=12,
                                            color=scatter_color,
                                            line=dict(width=2, color="#0F1117"),
                                            opacity=0.8
                                        )
                                    ))
                            
                            # PIE/Donut CHART
                            elif chart_type == "pie":
                                if len(y_axes) >= 1:
                                    pie_data = pdf.groupby(x_axis)[y_axes[0]].sum().reset_index()
                                    fig = go.Figure(data=[go.Pie(
                                        labels=pie_data[x_axis],
                                        values=pie_data[y_axes[0]],
                                        hole=0.4,
                                        textinfo="label+percent",
                                        textfont_size=12,
                                        marker=dict(colors=color_seq[:len(pie_data)], line=dict(color='#0F1117', width=2))
                                    )])
                                    fig.update_layout(
                                        annotations=[dict(
                                            text=f"Total<br>{pie_data[y_axes[0]].sum():,.0f}", 
                                            x=0.5, y=0.5, 
                                            font_size=16, 
                                            showarrow=False, 
                                            font_color="#E4E6F0"
                                        )]
                                    )
                            
                            # COMBO CHART
                            elif chart_type == "combo":
                                if len(y_axes) >= 2:
                                    # First metric as bar
                                    bar_color = color_seq[0]
                                    fig.add_trace(go.Bar(
                                        x=pdf[x_axis],
                                        y=pdf[y_axes[0]],
                                        name=f"{y_axes[0]} (Bar)",
                                        marker_color=bar_color,
                                        text=pdf[y_axes[0]].round(2) if show_data_labels else None,
                                        textposition="outside"
                                    ))
                                    # Second metric as line
                                    line_color = color_seq[1]
                                    fig.add_trace(go.Scatter(
                                        x=pdf[x_axis],
                                        y=pdf[y_axes[1]],
                                        name=f"{y_axes[1]} (Line)",
                                        mode="lines+markers",
                                        line=dict(width=3, color=line_color),
                                        yaxis="y2"
                                    ))
                                    fig.update_layout(
                                        yaxis2=dict(
                                            title=y_axes[1],
                                            overlaying="y",
                                            side="right",
                                            showgrid=False
                                        )
                                    )
                                else:
                                    st.warning("Combo chart requires at least 2 metrics. Using bar chart instead.")
                                    bar_color = color_seq[0]
                                    fig.add_trace(go.Bar(
                                        x=pdf[x_axis],
                                        y=pdf[y_axes[0]],
                                        name=y_axes[0],
                                        marker_color=bar_color
                                    ))
                            
                            # Global Layout Configuration
                            fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif", color="#E4E6F0", size=12),
                                title=dict(
                                    text=f"{plan.get('title', 'Analysis')}",
                                    font=dict(size=20, color="#E4E6F0"),
                                    x=0.5,
                                    xanchor="center"
                                ),
                                xaxis=dict(
                                    title=dict(text=x_axis, font=dict(color="#8B92A8")),
                                    gridcolor="#2D3142" if show_grid else "rgba(0,0,0,0)",
                                    linecolor="#2D3142",
                                    tickfont=dict(color="#8B92A8")
                                ),
                                yaxis=dict(
                                    title=dict(text=", ".join(y_axes), font=dict(color="#8B92A8")),
                                    gridcolor="#2D3142" if show_grid else "rgba(0,0,0,0)",
                                    linecolor="#2D3142",
                                    tickfont=dict(color="#8B92A8"),
                                    zeroline=False
                                ),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.3,
                                    xanchor="center",
                                    x=0.5,
                                    bgcolor="rgba(22, 25, 34, 0.8)",
                                    bordercolor="#2D3142",
                                    borderwidth=1,
                                    font=dict(size=11)
                                ),
                                margin=dict(l=60, r=60, t=80, b=100),
                                hovermode="x unified",
                                hoverlabel=dict(
                                    bgcolor="#1A1D29",
                                    bordercolor="#2D3142",
                                    font_size=12,
                                    font_family="Inter, sans-serif"
                                )
                            )
                            
                            # Render Chart
                            st.plotly_chart(fig, use_container_width=True,height=600, key="main_viz")
                            
                            
                        except Exception as viz_err:
                            st.error("Visualization generation failed.")
                            st.exception(viz_err)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Query returned no data. Try adjusting your question.")
