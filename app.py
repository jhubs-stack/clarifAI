import streamlit as st
from backend import load_index, search_faq, generate_response
import os

# Load ClarifAI's memory (index + faqs)
index, faqs = load_index()

st.set_page_config(
    page_title="ClarifAI FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)

# Inject custom CSS for a modern look
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .chat-bubble {
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 80%;
        }
        .user-msg {
            background-color: #DCF8C6;
            margin-left: auto;
            text-align: right;
        }
        .bot-msg {
            background-color: #F1F0F0;
            margin-right: auto;
        }
        .source {
            font-size: 0.85em;
            color: #888;
        }
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .sidebar-title {
            font-size: 1.4em;
            font-weight: bold;
            margin-top: 1rem;
        }
        .sidebar-sub {
            font-size: 0.9em;
            color: #666;
            margin-top: 0.5rem;
        }
        .suggested {
            font-size: 0.85em;
            color: #444;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with ClarifAI info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=120)
    st.markdown("<div class='sidebar-title'>ClarifAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>An AI-powered FAQ assistant using FAISS and mock embeddings.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='suggested'>
        <strong>Try asking:</strong><br>
        • How do I reset my password?<br>
        • What is your return policy?
    </div>
    <div class='sidebar-sub' style='margin-top: 1rem;'>
        🔐 This app runs offline using mock data.
    </div>
    """, unsafe_allow_html=True)

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("💬 ClarifAI")
st.caption("Ask a question from the FAQ. I'll find the best match!")

# Text input and Send button
with st.form(key="chat_form"):
    user_input = st.text_input("Your question", placeholder="e.g. Can I return an item?", key="input")
    submitted = st.form_submit_button("Send")

# Handle user message
if submitted and user_input:
    match = search_faq(user_input, index, faqs)
    response = generate_response(user_input, match)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("ClarifAI", response, match["question"]))

    # ✅ Clear the input field
    st.session_state["input"] = ""

# Display chat history using styled bubbles
for entry in st.session_state.chat_history:
    speaker = entry[0]
    message = entry[1]
    source = entry[2] if len(entry) > 2 else None

    is_user = speaker == "You"
    avatar = "🧑" if is_user else "🤖"
    alignment = "flex-end" if is_user else "flex-start"
    bubble_color = "#DCF8C6" if is_user else "#F1F0F0"
    text_align = "right" if is_user else "left"

    st.markdown(f"""
    <div style="display: flex; justify-content: {alignment}; margin-bottom: 10px;">
        <div style="
            background-color: {bubble_color};
            padding: 10px 15px;
            border-radius: 12px;
            max-width: 75%;
            text-align: {text_align};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 1rem;
        ">
            <div style="font-size: 1.2rem;">{avatar}</div>
            <div>{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if source:
        st.markdown(f"<div style='font-size:0.8em; color:gray; text-align:{text_align}; margin-bottom: 20px;'>📌 Matched FAQ: {source}</div>", unsafe_allow_html=True)
