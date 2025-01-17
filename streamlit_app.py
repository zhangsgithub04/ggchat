import streamlit as st
import google.generativeai as ggi

# Load API key from secrets
fetchheed_api_key = st.secrets["gemini_api_key"]
ggi.configure(api_key=fetchheed_api_key)

# Initialize model and chat
model = ggi.GenerativeModel("gemini-pro")
chat = model.start_chat()

def llm_response(question):
    """Send a message to the chat and return the response."""
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    st.title("Chat Application using Gemini Pro")

    # Store initial user input
    initial_input = None

    # Text input for user question
    user_quest = st.text_input("Ask a question:")

    # Button to trigger the question
    btn = st.button("Ask")

    if btn and user_quest:
        nonlocal initial_input

        # Check if it's the initial input
        if initial_input is None:
            initial_input = user_quest
            result = llm_response(user_quest)
            if result:
                st.subheader("Response:")
                for word in result:
                    st.text(word.text)
        else:
            # Check if subsequent input is related to initial input
            if initial_input.lower() not in user_quest.lower():
                st.warning("Please ask a follow-up question related to your initial query.")
                return

            result = llm_response(user_quest)
            if result:
                st.subheader("Response:")
                for word in result:
                    st.text(word.text)

if __name__ == "__main__":
    main()
