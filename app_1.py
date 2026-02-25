import streamlit as st
import openai
import logging
import time
import uuid
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

# =====================================
# CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Conversational Chatbot",
    page_icon="🌍",
    layout="centered"
)

MODEL_NAME = "gpt-3.5-turbo"
MAX_MESSAGES = 20
REQUEST_TIMEOUT = 30

# =====================================
# LOGGING SETUP
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AIChatbot")

# =====================================
# SECRETS VALIDATION
# =====================================

required_secrets = ["OPENAI_API_KEY", "MONGODB_URI"]

for secret in required_secrets:
    if secret not in st.secrets:
        st.error(f"Missing required secret: {secret}")
        st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# =====================================
# MONGODB CONNECTION
# =====================================

try:
    mongo_client = MongoClient(
        st.secrets["MONGODB_URI"],
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000,
        retryWrites=True
    )
    db = mongo_client["ai_chatbot_db"]
    conversations_collection = db["conversations"]

    # Indexes for performance
    conversations_collection.create_index(
        [("session_id", ASCENDING), ("created_at", ASCENDING)]
    )

    logger.info("MongoDB connected successfully")

except ConnectionFailure:
    logger.error("MongoDB connection failed", exc_info=True)
    st.error("Database connection failed.")
    st.stop()

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = "You are a helpful assistant."

# =====================================
# UI HEADER
# =====================================

st.title("🌍 AI Conversational Chatbot")
st.caption("Powered by OpenAI")

# =====================================
# SESSION MANAGEMENT
# =====================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Load existing conversation from MongoDB
    stored_messages = list(
        conversations_collection.find(
            {"session_id": st.session_state.session_id}
        ).sort("created_at", 1)
    )

    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in stored_messages
    ]

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# HELPER FUNCTIONS
# =====================================

def build_messages():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(st.session_state.messages)
    return messages


def trim_history():
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]


def save_message(role, content, token_usage=None):
    conversations_collection.insert_one({
        "session_id": st.session_state.session_id,
        "role": role,
        "content": content,
        "token_usage": token_usage,
        "created_at": datetime.utcnow()
    })


def generate_response(messages):
    try:
        start_time = time.time()

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            request_timeout=REQUEST_TIMEOUT
        )

        duration = time.time() - start_time
        tokens = response["usage"]["total_tokens"]

        logger.info(
            f"Response time: {duration:.2f}s | Tokens used: {tokens}"
        )

        return response["choices"][0]["message"]["content"].strip(), tokens

    except openai.error.RateLimitError:
        logger.warning("Rate limit exceeded.")
        st.error("Too many requests. Please try again shortly.")
        return None, None

    except openai.error.Timeout:
        logger.error("Request timed out.")
        st.error("Request timed out. Please try again.")
        return None, None

    except Exception:
        logger.error("Unexpected OpenAI error.", exc_info=True)
        st.error("AI service temporarily unavailable.")
        return None, None


# =====================================
# USER INPUT
# =====================================

user_input = st.chat_input("Type your message...")

if user_input:

    if len(user_input) > 2000:
        st.error("Input too long. Please shorten your message.")
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    save_message("user", user_input)

    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        reply, tokens = generate_response(build_messages())

    if reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        save_message("assistant", reply, tokens)

        st.chat_message("assistant").markdown(reply)

    trim_history()

# =====================================
# DISCLAIMER
# =====================================

st.divider()
st.warning(
    "⚠️ AI chatbot responses are generated automatically and may not always "
    "be accurate, complete, or up to date. Please verify critical information "
    "independently before making decisions. By using this chat, you agree to "
    "these terms."
)
