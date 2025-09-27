import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

intents = json.load(open('intents.json'))
patterns = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    #break
# print(len(patterns))
# print(len(tags))

vector = TfidfVectorizer() # to extract features from text
patterns_scale = vector.fit_transform(patterns) # fit and transform the patterns

# Create model
model = LogisticRegression(max_iter=10000)
model.fit(patterns_scale, tags)

def ChatBot(input_message):
    input_message = vector.transform([input_message])
    result = model.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == result:
            response = random.choice(intent['responses'])
            return response

# input_message = 'How can I contact you?'
# print(ChatBot(input_message))

st.title("University ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"AI ChatBot: " + ChatBot(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})