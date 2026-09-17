import streamlit as st
from openai import OpenAI
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
import os
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
st.title("Lab 4 - Course information chatbot")



#pull the key from Streamlit secrets.
if "openai_api_key" not in st.secrets:
    st.error("Please add your openai_api_key to secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])
MODEL = "gpt-4o-mini"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["openai_api_key"],
    model_name="text-embedding-3-small",
)

#create chromaDB client
Chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')


def create_lab4_collection():
    if 'Lab4_VectorDB' not in st.session_state:
        collection = Chroma_client.get_or_create_collection(
            "Lab4Collection", embedding_function=openai_ef
        )
        pdf_folder = "./pdfs"

        if os.path.isdir(pdf_folder):
            for filename in os.listdir(pdf_folder):
                filepath = os.path.join(pdf_folder, filename)
                if not os.path.isfile(filepath) or not filename.lower().endswith(".pdf"):
                    continue

                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                if text.strip():
                    collection.upsert(
                        documents=[text],
                        metadatas=[{"filename": filename}],
                        ids=[filename],
                    )

        st.session_state.Lab4_VectorDB = collection

    return st.session_state.Lab4_VectorDB


collection = create_lab4_collection()

# AI Prompts
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a friendly chatbot. Explain everything simply, the way you "
        "would explain it to a curious 10 year old: short sentences, no jargon, "
        "use everyday examples. "
        "Conversation flow rules:\n"
        "1. When the user asks a new question, answer it simply, then ask "
        "exactly: 'Do you want more info?'\n"
        "2. If the user says something like 'yes', give MORE detail on the "
        "same topic (still simple, still for a 10 year old), then ask "
        "'Do you want more info?' again.\n"
        "3. If the user says something like 'no', stop giving more detail on "
        "that topic and instead ask: 'Okay! What else can I help you with?'\n"
        "Always follow this pattern.\n"
        "4. If you are given course document excerpts to help answer the "
        "question, you MUST clearly say so at the start of your answer, "
        "e.g., 'Based on the course materials, ...'. If no relevant "
        "documents were retrieved, answer from your general knowledge instead."
    ),
}

# History promt 

if "messages" not in st.session_state:
    # messages holds the FULL history (system prompt + every user/assistant turn)
    st.session_state.messages = [SYSTEM_PROMPT]


# sidebar controls

st.sidebar.header("Buffer settings")
buffer_mode = st.sidebar.radio(
    "Choose a buffering strategy",
    ["Last 2 exchanges", "Token-based"],
    help="Part B.3 = 'Last 2 exchanges'. Part B.4 = 'Token-based'.",
)

max_tokens_allowed = st.sidebar.number_input(
    "max_tokens (for token-based buffer)",
    min_value=100,
    max_value=8000,
    value=1000,
    step=100,
)

#chat history 

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Buffer help

def build_last_two_exchanges_buffer(all_messages):
    """
    Part B.3: keep only the last two user messages and the two LLM responses
    that answered them, PLUS the system prompt (always kept, per Part C hint).
    """
    system_msgs = [m for m in all_messages if m["role"] == "system"]
    convo = [m for m in all_messages if m["role"] != "system"]

    # Each "exchange" is one user message + one assistant message.
    # Walk backwards and grab the last 2 complete exchanges' worth of turns.
    trimmed = convo[-4:]  # last 2 user + last 2 assistant messages (at most)

    return system_msgs + trimmed


def count_tokens(text, model=MODEL):
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # Fallback rough estimate if tiktoken isn't installed: ~4 chars per token
    return max(1, len(text) // 4)


def build_token_based_buffer(all_messages, max_tokens):
    """
    Part B.4: keep the system prompt always, then add messages from the
    MOST RECENT backwards until adding the next one would exceed max_tokens.
    """
    system_msgs = [m for m in all_messages if m["role"] == "system"]
    convo = [m for m in all_messages if m["role"] != "system"]

    running_total = sum(count_tokens(m["content"]) for m in system_msgs)
    kept = []

    # Walk from the newest message backwards
    for m in reversed(convo):
        t = count_tokens(m["content"])
        if running_total + t > max_tokens:
            break
        kept.insert(0, m)
        running_total += t

    return system_msgs + kept


def get_buffered_messages():
    if buffer_mode == "Last 2 exchanges":
        return build_last_two_exchanges_buffer(st.session_state.messages)
    else:
        return build_token_based_buffer(
            st.session_state.messages, max_tokens_allowed
        )


# chat inputs
if prompt := st.chat_input("Ask me anything..."):
    # 1. Save + show the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG retrieval: search the vector DB with the user's question
    rag_results = st.session_state.Lab4_VectorDB.query(
        query_texts=[prompt],
        n_results=3
    )
    retrieved_docs = rag_results["documents"][0]
    retrieved_filenames = rag_results["ids"][0]

    context_text = "\n\n".join(
        f"[{name}]\n{doc[:1000]}"
        for name, doc in zip(retrieved_filenames, retrieved_docs)
    )

    rag_context_msg = {
        "role": "system",
        "content": (
            "Here are relevant excerpts from course syllabi that may help "
            "answer the user's question:\n\n" + context_text
        ),
    }

    # 3. Build the buffered message list, with RAG context injected
    messages_to_send = get_buffered_messages() + [rag_context_msg]

    # 4. Stream the response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    # 5. Save the assistant's full response into history
    st.session_state.messages.append({"role": "assistant", "content": response})