# pip install streamlit langchain langchain-openai

import streamlit as st

def get_response(user_input):
    return "Saya tidak tahu"

# app config
st.set_page_config(page_title="Chat with websites")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# streamlit run src/app.py to run the GUI

# user input
user_query = st.chat_input("Tulis pesan anda di sini...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    with st.chat_message("Human"):
        st.write(user_query)
    
    with st.chat_message("AI"):
        st.write(response)



