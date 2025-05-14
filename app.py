import streamlit as st
from backend import load_index, search_faq, generate_response
import os
import time

# Load ClarifAI's memory (index + faqs)
index, faqs = load_index()

st.set_page_config(
    page_title="ClarifAI FAQ Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar with ClarifAI info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=120)
    st.markdown("## ClarifAI")
    st.markdown("An AI-powered FAQ assistant using FAISS and mock embeddings.")
    st.markdown("""
    **Try asking:**
    - How do I reset my password?
    - What is your return policy?

    🔐 This app runs offline using mock data.
    """)

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset input field if flagged
if st.session_state.get("reset_input"):
    st.session_state["input"] = ""
    st.session_state["reset_input"] = False

# Main container for input and chat
st.markdown("""
<div style="max-width: 700px; margin: auto;">
""", unsafe_allow_html=True)

# Header
st.title("💬 ClarifAI")
st.caption("Ask a question from the FAQ. I'll find the best match!")

# Text input and Send button
with st.form(key="chat_form"):
    user_input = st.text_input("Your question", placeholder="e.g. Can I return an item?", key="input", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

# Handle user message
if submitted and user_input:
    match = search_faq(user_input, index, faqs)

    # Typing animation placeholder
    with st.spinner("ClarifAI is typing..."):
        time.sleep(1.2)
        response = generate_response(user_input, match)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("ClarifAI", response, match["question"]))
    st.session_state["reset_input"] = True
    st.rerun()

# Display chat history
for entry in st.session_state.chat_history:
    speaker = entry[0]
    message = entry[1]
    source = entry[2] if len(entry) > 2 else None

    if speaker == "You":
        st.markdown(f"""
        <div style="clear: both;">
            <div style="
                float: right;
                background-color: #DCF8C6;
                padding: 14px 18px;
                border-radius: 10px;
                max-width: 100%;
                text-align: left;
                margin: 10px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="clear: both;">
            <div style="
                float: left;
                background-color: #F1F0F0;
                padding: 14px 18px;
                border-radius: 10px;
                max-width: 100%;
                text-align: left;
                margin: 10px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if source:
            st.markdown(f"<div style='clear: both; font-size: 0.85em; color: #888; margin: 4px 0 12px;'>📌 Matched FAQ: {source}</div>", unsafe_allow_html=True)

st.markdown("""</div>""", unsafe_allow_html=True)

# Auto-scroll to bottom
st.components.v1.html("""
<script>
    const anchor = window.parent.document.querySelector("iframe").contentWindow.document.body;
    anchor.scrollIntoView({ behavior: "smooth", block: "end" });
</script>
""", height=0)
