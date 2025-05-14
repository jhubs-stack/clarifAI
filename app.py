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

# Sidebar
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

# Main container for chat interface
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 1rem;
    }
    .chat-bubble {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-size: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .chat-user {
        background-color: #f7f7f8;
    }
    .chat-bot {
        background-color: #ffffff;
    }
    .chat-row {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .chat-avatar {
        border-radius: 50%;
        width: 32px;
        height: 32px;
        object-fit: cover;
    }
</style>
<div class="chat-container">
""", unsafe_allow_html=True)

# Display chat history
for entry in st.session_state.chat_history:
    speaker = entry[0]
    message = entry[1]
    source = entry[2] if len(entry) > 2 else None

    avatar_url = (
        "https://img.icons8.com/fluency/48/chatbot.png"
        if speaker == "ClarifAI"
        else "https://img.icons8.com/ios-filled/50/000000/user.png"
    )
    bubble_class = "chat-bot" if speaker == "ClarifAI" else "chat-user"

    st.markdown(f"""
    <div class="chat-row">
        <img src="{avatar_url}" class="chat-avatar">
        <div class="chat-bubble {bubble_class}" style="max-width: 700px;">
            <strong>{speaker}</strong><br>
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if source and speaker == "ClarifAI":
        st.markdown(f"<div style='margin-left: 44px; font-size: 0.85em; color: #888;'>📌 Matched FAQ: {source}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input box
with st.form(key="chat_input_form"):
    st.markdown("""<div style='max-width: 800px; margin: auto;'>""", unsafe_allow_html=True)
    user_input = st.text_input("Your question", key="input", label_visibility="collapsed", placeholder="e.g. Can I return an item?")
    st.form_submit_button("Send")
    st.markdown("</div>", unsafe_allow_html=True)

# Handle submission
if user_input:
    match = search_faq(user_input, index, faqs)
    with st.spinner("ClarifAI is typing..."):
        time.sleep(1.2)
        response = generate_response(user_input, match)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("ClarifAI", response, match["question"]))
    st.session_state["reset_input"] = True
    st.rerun()

# Auto-scroll
st.components.v1.html("""
<script>
    const anchor = window.parent.document.querySelector("iframe").contentWindow.document.body;
    anchor.scrollIntoView({ behavior: "smooth", block: "end" });
</script>
""", height=0)
