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
st.set_page_config(page_title="KPI Chatbot", page_icon="💬", layout="wide")
st.title("💬 KPI Chatbot")
st.caption("Ask questions about your dataset in plain English.")

# ── Dynamically read schema from whatever DB is loaded ───────────────────────
@st.cache_data
def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]

        # Get sample values for text columns (top 5 distinct)
        col_details = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            col_details.append(f"  {col_name:<25} {col_type}")

            # For text columns, peek at distinct sample values
            if col_type.upper() in ("TEXT", "VARCHAR", "CHAR", ""):
                try:
                    cursor.execute(
                        f"SELECT DISTINCT {col_name} FROM {table} "
                        f"WHERE {col_name} IS NOT NULL LIMIT 5"
                    )
                    samples = [str(r[0]) for r in cursor.fetchall()]
                    if samples:
                        col_details[-1] += f"   -- e.g. {', '.join(samples)}"
                except Exception:
                    pass

        schema_parts.append(
            f"Table: {table}  ({row_count:,} rows)\n" + "\n".join(col_details)
        )

    conn.close()
    return "\n\n".join(schema_parts)


def build_system_prompt(schema: str) -> str:
    return f"""You are a data analyst assistant. Your job is to answer the user's \
question by writing a single SQLite SQL query against the database described below.

DATABASE SCHEMA:
{schema}

Rules:
- Always respond with ONLY a valid SQLite SQL query. Nothing else.
- Do not include markdown, backticks, or any explanation — just the raw SQL.
- For top-N questions, use LIMIT N.
- Always alias computed columns with a readable name (e.g. AVG(rating) AS avg_rating).
- If a column might contain NULLs, filter them out when aggregating.
- Infer column meaning from the schema samples above — do not assume anything not shown.
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
def get_sql(question: str, history: list, system_prompt: str) -> str:
    messages = history + [{"role": "user", "content": question}]
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=system_prompt,
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
    st.session_state.sql_history = []

# Build system prompt once per session from live DB schema
schema = get_schema(DB_PATH)
SYSTEM_PROMPT = build_system_prompt(schema)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Try asking...")
    example_questions = [
        "What is the average rating per main category?",
        "Top 5 most discounted products",
        "Which category has the highest average price?",
        "How many products have a rating above 4.3?",
        "What is the average discount percentage across all products?",
        "Show me products with more than 50,000 ratings",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    # Show live DB info instead of hardcoded caption
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    for t in tables["name"]:
        count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", conn)["n"][0]
        st.caption(f"Table: `{t}` · {count:,} rows")
    conn.close()

    with st.expander("View schema"):
        st.code(schema, language="sql")

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
                sql = get_sql(user_input, st.session_state.sql_history, SYSTEM_PROMPT)

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