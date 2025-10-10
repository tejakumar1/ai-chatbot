import streamlit as st
import opik
from opik import track, opik_context, Opik 
import os
import uuid 
import time

# --- Third-Party LLM & AGENT Imports (Updated) ---
from langchain_openai import AzureChatOpenAI 
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_experimental.tools.python.tool import PythonREPLTool # ✅ Replaces Calculator
from langchain_core.tools import Tool
# ---------------------------------------------------

# --- 1. Opik and Azure Environment Setup (Unchanged) ---

# AZURE KEYS 
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "726WoJruf3CtnvJbpF4YQarspGo2wvAWQv2dGc8J6Ihx4mPIKM2qJQQJ99BDACHYHv6XJ3w3AAAAACOGIsap")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://jon-m900oi9e-eastus2.cognitiveservices.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
GPT_DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME", "gpt-4o-mini") 

# OPIK KEYS
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE_NAME", "azure-chat-dashboard-opik")
OPIK_PROJECT = os.getenv("OPIK_PROJECT_NAME", "Azure-Chat-Dashboard-Opik")

# Initialize Opik client globally
opik_client = None
try:
    opik_client = Opik()
    st.sidebar.success(f"Opik client connected. Target: {OPIK_WORKSPACE}/{OPIK_PROJECT}")
except Exception as e:
    st.sidebar.error(f"FATAL: Opik client failed to initialize. Check OPIK_API_KEY. Error: {e}")

# --- Initialize LLM and Agent Components ---
llm = None
agent_executor = None
tools = []

try:
    llm = AzureChatOpenAI( 
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=GPT_DEPLOYMENT_NAME,
        temperature=0, 
        streaming=False
    )
    st.sidebar.caption(f"LLM: {GPT_DEPLOYMENT_NAME} connected.")

    # ✅ Use PythonREPLTool for calculator-like functionality
    tools = [PythonREPLTool()]
    
    # 2. Get the official Agent Prompt
    prompt_template = hub.pull("hwchase17/openai-functions-agent")
    
    # 3. Create the Agent
    agent = create_openai_functions_agent(llm, tools, prompt_template)

    # 4. Create the Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    st.sidebar.caption("Agent Executor initialized with Python REPL tool.")

except Exception as e:
    st.sidebar.error(f"FATAL: Agent/LLM failed to initialize. Error: {e}")


# --- 2. Opik-Tracked Agent Function ---
@track(project_name=OPIK_PROJECT)
def generate_llm_response(prompt: str, thread_id: str) -> tuple[str, str | None]:
    """
    Calls the Agent Executor with the user's prompt.
    """
    llm_response = "Error: Agent or LLM not initialized."
    trace_url = None
    
    time.sleep(0.5)
    
    if agent_executor:
        try:
            response = agent_executor.invoke({"input": prompt}) 
            llm_response = response["output"] 
        except Exception as e:
            llm_response = f"Agent Execution Error: {e}"

    # --- Opik Tracing Logic ---
    try:
        current_trace_data = opik_context.get_current_trace_data()
        trace_id = current_trace_data.id 
        project_name = current_trace_data.project_name if current_trace_data.project_name else OPIK_PROJECT
        opik_base_url = "https://www.comet.com/opik"
        trace_url = f"{opik_base_url}/{OPIK_WORKSPACE}/projects/{project_name}/traces/{trace_id}"
        
    except Exception as e:
        st.warning(f"⚠️ Tracing Error: Could not generate trace URL. Error: {e}")

    return llm_response, trace_url

# ---------------------------------------------------------------------

# --- 3. Streamlit Application ---

st.set_page_config(page_title="Azure Opik Agent Chatbot", layout="centered")
st.title("Azure Agent (Traced by Opik) Chat 🤖")
st.caption(f"Traces logged to Opik Workspace: **{OPIK_WORKSPACE}** / Project: **{OPIK_PROJECT}**")

# Initialize chat history and unique session/thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "opik_thread_id" not in st.session_state:
    st.session_state.opik_thread_id = str(uuid.uuid4())
    st.info(f"Opik Thread ID: `{st.session_state.opik_thread_id[:8]}...` (Groups conversation turns)")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace_url"):
            st.markdown("---") 
            st.markdown(f"**🔬 Trace:** [View in Opik]({message['trace_url']})")

# React to user input
if prompt := st.chat_input("Ask a question (e.g., What is 123*45?)"):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Get LLM Response and Opik Trace URL
    llm_response, trace_link = generate_llm_response(prompt, st.session_state.opik_thread_id)
    
    # 3. Flush Opik client to send trace data
    if opik_client:
        try:
            opik_client.flush()
        except Exception as e:
            st.warning(f"Could not flush Opik client. Error: {e}")

    # 4. Display Assistant Message
    with st.chat_message("assistant"):
        st.markdown(llm_response)
        if trace_link:
            st.markdown("---") 
            st.markdown(f"**🔬 Trace:** [View in Opik]({trace_link})")
            
    # 5. Save assistant response and trace URL to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": llm_response,
        "trace_url": trace_link
    })
