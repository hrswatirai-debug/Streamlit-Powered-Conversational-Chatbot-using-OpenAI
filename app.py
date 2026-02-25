import streamlit as st
import openai
import logging
import time
from datetime import datetime

# ======================================
# CONFIGURATION
# ======================================

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered"
)

MAX_MESSAGES = 20
REQUEST_TIMEOUT = 30
MODEL_NAME = "gpt-3.5-turbo"

# ======================================
# LOGGING SETUP
# ======================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MedicalAI")

# ======================================
# OPENAI CLIENT INITIALIZATION
# ======================================

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not configured.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ======================================
# SYSTEM PROMPT (STRICT MEDICAL GUARDRAILS)
# ======================================

SYSTEM_PROMPT = """
You are a medical assistant chatbot for educational purposes only.

You have access to previous messages in this session.
Use them for consistency.

Strict rules:
- Provide general medical education only.
- Do NOT diagnose medical conditions.
- Do NOT prescribe medications or provide dosages.
- Do NOT replace professional medical consultation.
- Always mention warning signs requiring urgent care.
- Encourage consulting a licensed healthcare professional.
- Be calm, empathetic, and professional.
"""

# ======================================
# UI HEADER
# ======================================

st.title("🩺 Medical AI Chatbot")
st.caption("Educational Use Only")

# ======================================
# SESSION STATE
# ======================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_start_time" not in st.session_state:
    st.session_state.conversation_start_time = datetime.utcnow()

# ======================================
# DISPLAY PREVIOUS MESSAGES
# ======================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================================
# HELPER FUNCTIONS
# ======================================

def build_messages():
    """Build OpenAI-compatible message structure."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(st.session_state.messages)
    return messages


def trim_history():
    """Keep only recent messages to control memory."""
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]


def generate_ai_response(messages):
    """Call OpenAI with retry-safe handling."""
    try:
        start_time = time.time()

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=500,
            request_timeout=REQUEST_TIMEOUT
        )

        duration = time.time() - start_time
        token_usage = response["usage"]["total_tokens"]

        logger.info(
            f"Response Time: {duration:.2f}s | Tokens Used: {token_usage}"
        )

        return response["choices"][0]["message"]["content"].strip()

    except openai.error.RateLimitError:
        logger.warning("Rate limit hit.")
        st.error("Too many requests. Please wait a moment and try again.")
        return None

    except openai.error.Timeout:
        logger.error("OpenAI request timed out.")
        st.error("Request timed out. Please try again.")
        return None

    except Exception:
        logger.error("Unexpected OpenAI error.", exc_info=True)
        st.error("AI service temporarily unavailable.")
        return None


# ======================================
# USER INPUT
# ======================================

user_input = st.chat_input("Ask a medical question...")

if user_input:

    # Basic validation
    if len(user_input) > 2000:
        st.error("Input too long. Please shorten your message.")
        st.stop()

    # Append user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    st.chat_message("user").markdown(user_input)

    with st.spinner("Analyzing your question..."):
        ai_reply = generate_ai_response(build_messages())

    if ai_reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_reply}
        )

        st.chat_message("assistant").markdown(ai_reply)

    trim_history()

# ======================================
# FOOTER DISCLAIMER
# ======================================

st.divider()
st.warning(
    "⚠️ This chatbot provides educational information only. "
    "It does NOT replace professional medical advice. "
    "Consult a licensed healthcare provider for diagnosis or treatment."
)
