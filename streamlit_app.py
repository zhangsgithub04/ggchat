import streamlit as st
import google.generativeai as ggi
from transformers import pipeline

# Load API key from secrets
fetchheed_api_key = st.secrets["gemini_api_key"]
ggi.configure(api_key=fetchheed_api_key)

# Initialize model and chat
model = ggi.GenerativeModel("gemini-pro")
chat = model.start_chat()

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification")

def llm_response(question):
    """Send a message to the chat and return the response."""
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def is_relevant(initial_input, subsequent_input):
    """Use zero-shot classifier to determine relevance."""
    candidate_labels = ["computing", "Linux", "cybersecurity", "operating system", "networking", "attack", "vulnerability"]
    result = classifier(subsequent_input, candidate_labels)
    if result["labels"][0] == "relevant":
        return True
    else:
        return False

def main():
    st.title("Chat Application using Gemini Pro")

    # Add instructions or context upfront
    st.markdown("**Instructions:**")
    st.markdown("* Ask a question related to a specific topic.")
    st.markdown("* The model will respond based on the context provided.")
    st.markdown("* Please ask follow-up questions related to the initial query.")

    # Initialize session state
    if "initial_input" not in st.session_state:
        st.session_state.initial_input = None

    # Text input for user question
    user_quest = st.text_input("Ask a question:")

    # Button to trigger the question
    btn = st.button("Ask")

    if btn and user_quest:
        # Check if it's the initial input
        if st.session_state.initial_input is None:
            st.session_state.initial_input = user_quest
            result = llm_response(user_quest)
            if result:
                st.subheader("Response:")
                for word in result:
                    st.text(word.text)
        else:
            # Check if subsequent input is relevant using zero-shot classifier
            if is_relevant(st.session_state.initial_input, user_quest):
                result = llm_response(user_quest)
                if result:
                    st.subheader("Response:")
                    for word in result:
                        st.text(word.text)
            else:
                st.warning("Please ask a follow-up question related to your initial query.")

if __name__ == "__main__":
    main()
