import streamlit as st
import random
import json
from utils import predict_intent

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Chatbot response logic
def chatbot_response(user_input):
    intent_tag = predict_intent(user_input)
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that. Could you please rephrase?"

# Streamlit UI
st.title("Advanced NLP Chatbot")



if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You: ", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state["messages"].append({"user": True, "text": user_input})
    response = chatbot_response(user_input)
    st.session_state["messages"].append({"user": False, "text": response})


for msg in st.session_state["messages"]:
    if msg["user"]:
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")