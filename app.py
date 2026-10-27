"""Run with: python -m streamlit run app.py"""
import os
import ssl
from pathlib import Path
from urllib.parse import urlsplit

import certifi
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI, DefaultHttpxClient, APIError

st.set_page_config(page_title="Azure AI Chat", page_icon="💬", layout="centered")
st.title("💬 Azure AI Chat")
st.caption("Your personal assistant, powered by Azure OpenAI")

# Read local edits on each rerun; hosting environment variables take precedence.
local_config = dotenv_values(Path(__file__).resolve().parent / ".env")
def setting(name):
    return (os.getenv(name) or local_config.get(name) or "").strip()

endpoint = setting("AZURE_OPENAI_ENDPOINT").rstrip("/")
api_key = setting("AZURE_OPENAI_API_KEY")
deployment = setting("AZURE_OPENAI_DEPLOYMENT")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Chat settings")
    st.caption("Messages are kept for this browser session.")
    max_tokens = st.slider("Maximum reply tokens", 100, 1500, 800, 100)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not all([endpoint, api_key, deployment]):
    st.info("Add AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_DEPLOYMENT to the .env file beside app.py.")
    st.stop()

parts = urlsplit(endpoint)
if parts.scheme != "https" or not parts.netloc or parts.query or parts.fragment:
    st.error("Use an HTTPS Azure resource endpoint without query parameters.")
    st.stop()
if parts.path.rstrip("/") not in ("", "/openai/v1"):
    st.error("Use your Azure resource URL or its /openai/v1/ URL, not a project or chat/completions URL.")
    st.stop()
base_url = endpoint + "/" if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1/"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything…")
if prompt and prompt.strip():
    prompt = prompt.strip()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        output = st.empty()
        output.markdown("Thinking…")
        reply = ""
        try:
            # An explicit CA bundle avoids the missing default certificate path.
            context = ssl.create_default_context(cafile=certifi.where())
            # Use the SDK's own HTTP client to match its installed transport version.
            with OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=DefaultHttpxClient(verify=context, timeout=60.0),
                max_retries=1,
            ) as client:
                history = st.session_state.messages[-20:]
                with client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "system", "content": "You are a helpful assistant. Keep answers clear and concise."}]
                    + history + [{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=max_tokens,
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            reply += chunk.choices[0].delta.content
                            output.markdown(reply + " ▌")
            if reply:
                output.markdown(reply)
                st.session_state.messages.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply},
                ])
            else:
                output.warning("No text was returned. Try rephrasing your question.")
        except APIError as error:
            print(
                "CONNECTION CAUSE:",
                type(error.__cause__).__name__,
                str(error.__cause__),
                flush=True,
            )
            output.empty()
            status = getattr(error, "status_code", None)
            explanations = {
                401: "Authentication failed. Check that the key and endpoint belong to the same Azure resource.",
                403: "Access denied. Check the resource's access and network settings.",
                404: "Endpoint or deployment not found. Check your working endpoint and exact deployment name in .env.",
                429: "Azure rate limit or quota reached. Wait before retrying and check your deployment quota.",
                400: "Azure rejected the request. Check that the deployment supports these chat parameters.",
            }
            st.error(explanations.get(status, "The request failed. Check your connection and Azure service availability."))
            st.caption(f"Error type: {type(error).__name__}; status: {status or 'connection error'}. Failed turns are not saved.")
        except (OSError, ssl.SSLError):
            output.empty()
            st.error("Could not load the certificate bundle. Run: python -m pip install --force-reinstall certifi")
