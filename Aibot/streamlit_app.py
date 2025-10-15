import os
import uuid
import time
import json
import streamlit as st
from dotenv import load_dotenv
from user_agents import parse as ua_parse
import pandas as pd
from opik import track, configure
import openai

# ---------------- Load Environment ----------------
load_dotenv()

# Azure OpenAI configuration
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
deployment_name = os.getenv("GPT_DEPLOYMENT_NAME", "gpt-4o-mini")

# ---------------- Configure Local OPIK ----------------
configure(use_local=True)
TRACE_FILE = "local_traces.json"

if os.path.exists(TRACE_FILE):
    with open(TRACE_FILE, "r") as f:
        saved_traces = json.load(f)
else:
    saved_traces = []

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Multi-Agent Dashboard", layout="wide")
st.title("🤖 AI Multi-Agent (Local Traces)")
st.caption("All traces are stored locally and viewable below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.info(f"Session ID: `{st.session_state.session_id[:8]}...`")

# ---------------- Helper Functions ----------------
def get_user_metadata():
    metadata = {}
    try:
        ua_string = st.runtime.scriptrunner.script_run_context.request.headers.get("user-agent", "")
        user_agent = ua_parse(ua_string)
        metadata["device"] = f"{user_agent.device.family} | {user_agent.os.family} | {user_agent.browser.family}"
    except Exception:
        metadata["device"] = "unknown"
    try:
        import requests
        ip_res = requests.get("https://ipinfo.io/json").json()
        metadata.update({
            "ip": ip_res.get("ip"),
            "city": ip_res.get("city"),
            "region": ip_res.get("region"),
            "country": ip_res.get("country")
        })
    except Exception:
        metadata.update({"ip":"unknown","city":"unknown","region":"unknown","country":"unknown"})
    return metadata

def python_calculator(code: str) -> str:
    try:
        from asteval import Interpreter
        aeval = Interpreter()
        return str(aeval(code))
    except Exception as e:
        return f"Calculator Error: {e}"

def file_reader(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"File Read Error: {e}"

def generate_response(prompt: str) -> str:
    """
    Generates a real AI response using Azure OpenAI Chat Completions.
    """
    try:
        response = openai.chat.completions.create(
            model=deployment_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI call error: {e}"

# ---------------- Display Previous Messages ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace_id"):
            st.markdown(f"🧠 Trace ID: `{msg['trace_id']}`")

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    metadata = get_user_metadata()
    metadata["session_id"] = st.session_state.session_id
    metadata["turn"] = len(st.session_state.messages)//2 + 1

    @track(project_name="local_project", metadata={"prompt": prompt, **metadata})
    def process_turn():
        response_texts = []

        if "calculate" in prompt.lower() or "math" in prompt.lower():
            response_texts.append(f"Calculator Result: {python_calculator(prompt)}")
        if "read file" in prompt.lower() or "open file" in prompt.lower():
            response_texts.append(f"File Content:\n{file_reader('example.txt')}")
        
        # Real AI-generated response
        response_texts.append(f"AI Response:\n{generate_response(prompt)}")
        return "\n\n".join(response_texts)

    combined_response = process_turn()
    trace_id = str(uuid.uuid4())[:8]

    with st.chat_message("assistant"):
        st.markdown(combined_response)
        st.markdown(f"🧠 Trace ID: `{trace_id}`")

    # Save traces
    trace_entry_user = {"role":"user","content":prompt,"trace_id":trace_id,"metadata":metadata,"timestamp":time.time()}
    trace_entry_assistant = {"role":"assistant","content":combined_response,"trace_id":trace_id,"metadata":metadata,"timestamp":time.time()}
    saved_traces.extend([trace_entry_user, trace_entry_assistant])
    st.session_state.messages.append({"role":"assistant","content":combined_response,"trace_id":trace_id})

    with open(TRACE_FILE, "w") as f:
        json.dump(saved_traces, f, indent=2)

# ---------------- Local Trace Dashboard ----------------
st.divider()
st.subheader("📊 Local Trace Dashboard (All Sessions)")

if saved_traces:
    df = pd.DataFrame(saved_traces)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["metadata"] = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})

    session_ids = df["metadata"].apply(lambda x: x.get("session_id", "unknown")).unique()
    sessions = ["All"] + sorted(session_ids)
    roles = ["All", "user", "assistant"]

    selected_session = st.selectbox("Filter by Session ID", sessions)
    selected_role = st.selectbox("Filter by Role", roles)
    search_text = st.text_input("Search messages...")

    filtered_df = df.copy()
    if selected_session != "All":
        filtered_df = filtered_df[filtered_df["metadata"].apply(lambda x: x.get("session_id", "unknown")) == selected_session]
    if selected_role != "All":
        filtered_df = filtered_df[filtered_df["role"] == selected_role]
    if search_text:
        filtered_df = filtered_df[filtered_df["content"].str.contains(search_text, case=False, na=False)]

    for idx, row in filtered_df.iterrows():
        with st.expander(f"{row['time']} | {row['role']} | Trace ID: {row['trace_id']}"):
            st.markdown(row["content"])
            st.markdown(f"Metadata: `{row['metadata']}`")
else:
    st.info("No traces recorded yet.")

# ---------------- Download Button ----------------
st.divider()
st.subheader("💾 Download All Traces")
if os.path.exists(TRACE_FILE):
    with open(TRACE_FILE, "r") as f:
        json_data = f.read()
    st.download_button("Download Traces as JSON", data=json_data, file_name="local_traces.json")
else:
    st.info("No trace file found yet.")









