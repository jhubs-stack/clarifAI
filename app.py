import streamlit as st
from backend import load_index, search_faq, generate_response

# Load FAISS index and FAQs
index, faqs = load_index()

# Page settings
st.set_page_config(
    page_title="ClarifAI FAQ Assistant",
    page_icon="💬",
    layout="centered"
)

# Inject custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
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
            background-color: #e0f0ff;
            margin-left: auto;
            text-align: right;
        }
        .bot-msg {
            background-color: #f0f0f0;
            margin-right: auto;
        }
        .source {
            font-size: 0.85em;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with ClarifAI info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=80)
    st.markdown("### 🤖 ClarifAI")
    st.markdown("A simple, local FAQ assistant powered by mock embeddings and FAISS.")
    st.markdown("**Try asking:**\n- How do I reset my password?\n- What is your return policy?")
    st.markdown("🔐 *Note: This is running offline with mock embeddings.*")

# Main app layout
st.title("💬 ClarifAI")
st.caption("Ask a question from the FAQ. I'll find the best match!")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Type your question", placeholder="e.g. Can I return an item?", key="input")

# Handle input
if user_input:
    match = search_faq(user_input, index, faqs)
    response = generate_response(user_input, match)

    # Add to history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("ClarifAI", response, match["question"]))

# Display conversation
for entry in st.session_state.chat_history:
    if entry[0] == "You":
        st.markdown(f"<div class='chat-bubble user-msg'>{entry[1]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble bot-msg'>{entry[1]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='source'>📌 Matched FAQ: {entry[2]}</div>", unsafe_allow_html=True)
