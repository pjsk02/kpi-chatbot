import os
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

# ── Load API key from secrets/.env ──────────────────────────────────────────
load_dotenv("secrets/.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DB_PATH = "amazon.db"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Amazon KPI Chatbot", page_icon="📦", layout="wide")
st.title("📦 Amazon KPI Chatbot")
st.caption("Ask questions about the Amazon product dataset in plain English.")

# ── Table schema (sent to Claude so it knows your data) ──────────────────────
TABLE_SCHEMA = """
Table name: products

Columns:
  product_id          TEXT     - unique product identifier
  product_name        TEXT     - full product name
  category            TEXT     - full category path (pipe-separated)
  main_category       TEXT     - top-level category (e.g. Electronics, Computers&Accessories)
  discounted_price    REAL     - selling price in INR (₹)
  actual_price        REAL     - original price before discount in INR (₹)
  discount_percentage REAL     - discount as a number (e.g. 64 means 64%)
  rating              REAL     - average rating out of 5
  rating_count        INTEGER  - number of ratings

Sample main_category values:
  Electronics, Computers&Accessories, Home&Kitchen, OfficeProducts, 
  MusicalInstruments, HealthPersonalCare, Toys&Games, Car&Motorbike
"""

SYSTEM_PROMPT = f"""You are a data analyst assistant for an Amazon product dataset.
Your job is to answer the user's question by writing a single SQLite SQL query.

{TABLE_SCHEMA}

Rules:
- Always respond with ONLY a valid SQLite SQL query. Nothing else.
- Do not include markdown, backticks, or explanation — just the raw SQL.
- Use main_category for category-level questions.
- Prices are in INR (Indian Rupees).
- For top-N questions, use LIMIT N.
- discount_percentage is already a number (64 = 64%), do not divide it.
- rating_count may be NULL for some rows — use WHERE rating_count IS NOT NULL when relevant.
- Always alias computed columns with a readable name (e.g. AVG(rating) AS avg_rating).
"""

# ── Helper: run SQL and return a dataframe ───────────────────────────────────
def run_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

# ── Helper: ask Claude to generate SQL ──────────────────────────────────────
def get_sql(question: str, history: list) -> str:
    messages = history + [{"role": "user", "content": question}]
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()

# ── Helper: ask Claude to explain results in plain English ───────────────────
def explain_results(question: str, sql: str, df: pd.DataFrame) -> str:
    summary = df.to_string(index=False) if len(df) <= 20 else df.head(10).to_string(index=False) + f"\n... ({len(df)} rows total)"
    prompt = f"""The user asked: "{question}"

You ran this SQL:
{sql}

Results:
{summary}

Write a short, clear, friendly answer in 2-3 sentences explaining what the data shows.
Use numbers from the results. Do not repeat the SQL. Do not use bullet points."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ── Helper: auto-select the best chart type ─────────────────────────────────
def render_chart(df: pd.DataFrame):
    if df.empty or len(df.columns) < 2:
        return

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return

    y_col = num_cols[0]
    x_col = text_cols[0] if text_cols else cols[0]

    # Truncate long product names for readability
    if df[x_col].dtype == object:
        df = df.copy()
        df[x_col] = df[x_col].astype(str).str[:40]

    if len(df) == 1:
        # Single value — metric card style
        st.metric(label=y_col.replace("_", " ").title(), value=f"{df[y_col].iloc[0]:,.2f}")
    elif len(df) <= 15:
        fig = px.bar(df, x=x_col, y=y_col,
                     labels={x_col: x_col.replace("_"," ").title(),
                              y_col: y_col.replace("_"," ").title()},
                     color_discrete_sequence=["#4f8ef7"])
        fig.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(df, x=x_col, y=y_col,
                      labels={x_col: x_col.replace("_"," ").title(),
                               y_col: y_col.replace("_"," ").title()})
        fig.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── Session state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []  # for multi-turn context

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Try asking...")
    example_questions = [
        "What is the average rating per main category?",
        "Top 5 most discounted products",
        "Which category has the highest average price?",
        "How many products have a rating above 4.5?",
        "What is the average discount percentage across all products?",
        "Show me products with more than 50,000 ratings",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    st.caption("Dataset: Amazon India Products · 1,465 rows")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sql_history = []
        st.rerun()

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg:
            st.dataframe(msg["dataframe"], use_container_width=True)
        if "chart_df" in msg:
            render_chart(msg["chart_df"].copy())
        if "sql" in msg:
            with st.expander("Generated SQL"):
                st.code(msg["sql"], language="sql")

# ── Chat input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask a question about the data...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Step 1: Generate SQL
                sql = get_sql(user_input, st.session_state.sql_history)

                # Step 2: Run query
                result_df = run_query(sql)

                # Step 3: Explain results
                explanation = explain_results(user_input, sql, result_df)

                # Step 4: Render
                st.markdown(explanation)
                if not result_df.empty:
                    st.dataframe(result_df, use_container_width=True)
                    render_chart(result_df.copy())
                with st.expander("Generated SQL"):
                    st.code(sql, language="sql")

                # Save to history
                st.session_state.sql_history.append({"role": "user", "content": user_input})
                st.session_state.sql_history.append({"role": "assistant", "content": sql})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": explanation,
                    "dataframe": result_df,
                    "chart_df": result_df,
                    "sql": sql,
                })

            except Exception as e:
                err = f"Something went wrong: `{e}`\n\nTry rephrasing your question."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})