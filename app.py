
import streamlit as st
from chatbot import get_answer


# Configure the webpage
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)


# Page heading
st.title("🎓 College FAQ Chatbot")
st.write(
    "Ask me questions about college admissions, "
    "fees, exams, certificates, and student services."
)

st.caption("Powered by Python, NLTK, TF-IDF and Cosine Similarity")


# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I am your College FAQ Chatbot. "
                "What would you like to know?"
            )
        }
    ]


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Input box for the user
user_question = st.chat_input("Type your question here...")


if user_question:
    # Display and store the user's question
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Find the answer
    try:
        answer, score = get_answer(user_question)

        response = (
            f"{answer}\n\n"
            f"**Similarity score:** {score:.2f}"
        )

    except Exception:
        response = (
            "Sorry, something went wrong while processing "
            "your question. Please try again."
        )

    # Display and store the chatbot's response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)


# Sidebar information
with st.sidebar:
    st.header("About this chatbot")

    st.write(
        "This chatbot finds the FAQ question most similar "
        "to the question you enter."
    )

    st.write("**Technologies used:**")
    st.write("- Python")
    st.write("- NLTK")
    st.write("- Pandas")
    st.write("- TF-IDF")
    st.write("- Cosine Similarity")
    st.write("- Streamlit")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
