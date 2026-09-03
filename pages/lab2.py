import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)
summary_type = st.sidebar.selectbox ( 
    'summarize document in', 
    ['100 words', '2 connecting paragraphs', '5 bullet points']
)

use_nano = st.sidebar.checkbox('Use GPT nano(faster)')
if use_nano:
    model = "gpt-5-nano"
else:
    model = "gpt-5-mini"

# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.openai_api_key


# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

if uploaded_file:
    # Process the uploaded file.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n Summarize this document in {summary_type}.",
        }
    ]
    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)