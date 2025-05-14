import streamlit as st
from backend import load_index, search_faq, generate_response

# Load ClarifAI's memory (index + faqs)
index, faqs = load_index()

st.set_page_config(page_title="ClarifAI FAQ Assistant", page_icon="💬")

st.title("💬 ClarifAI")
st.subheader("Ask me anything from the FAQ")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Your question", key="input")

if user_input:
    match = search_faq(user_input, index, faqs)
    answer = generate_response(user_input, match)

    # Save to history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("ClarifAI", answer))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(msg)
