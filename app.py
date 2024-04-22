import math
import os
from datetime import datetime

import openai
import PyPDF2
import streamlit as st
from openai import OpenAI

from helper.utils import *

st.set_page_config(layout="wide", page_title="Document Search using QIM🤖📖")
st.header("Document Search using Quantized Influence Measure (QIM)🤖📖")
st.write("---")


# Streamlit sidebar setup for user interface
with st.sidebar:
    # Create an expandable instruction manual section in the sidebar
    with st.expander("Instruction Manual 📖"):
        # Display the instruction manual for the Document Data Chatbot in a formatted markdown
        st.markdown(
            """
            # Document Search App Instruction Manual 📖🤖
            
            Welcome to the Document Search App! This guide will help you quickly start using the app to find information in your documents.
            
            ## Quick Start Guide
            
            1. **Upload Document**: Click on the "Upload documents" button in the sidebar and select your PDF or text files. Multiple files can be uploaded at once.
            2. **Enter Keywords**: After your documents are uploaded, use the chat input at the bottom of the app to type your query. For example, you could type keywords or questions related to the content you're interested in.
            3. **Review Results**: Hit 'Enter' to submit your query. The app will process your input and display the most relevant information from your documents in the form of a table right within the chat interface.
            
            ## Credits
            
            This app was created by Yiqiao Yin. For more about his work, visit his [website](https://www.y-yin.io/) or connect with him on [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/).
            
            Thank you for using the Document Search App! We hope it serves your information retrieval needs effectively. 🚀📈
            """
        )

    # File uploader widget allowing users to upload text and PDF documents
    uploaded_files = st.file_uploader(
        "Upload documents", accept_multiple_files=True, type=["txt", "pdf"]
    )

    # Inform the user how many documents have been loaded
    st.success(f"{len(uploaded_files)} document(s) loaded...")

    # Chunk size
    chunk_size_input = st.number_input(
        "Insert an integer (for size of chunks, i.e. 2 means 2 sentences a chunk):",
        value=2,
        step=1,
    )

    # Quantization
    q_levels = st.number_input(
        "Insert an integer for levels of quantization:",
        value=2,
        step=1,
        min_value=2,
        max_value=31,
    )

    # Input filter
    top_n = st.number_input(
        "Insert a number (top n rows to be selected):", value=3, step=1
    )

    # Clear button
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Credit
    current_year = current_year()  # This will print the current year
    st.markdown(
        f"""
            <h6 style='text-align: left;'>Copyright © 2010-{current_year} Present Yiqiao Yin</h6>
        """,
        unsafe_allow_html=True,
    )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Reset everything
if clear_button:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Check if any files have been uploaded
if uploaded_files is None:
    # Display a message prompting the user to upload files
    st.info("Upload files to analyze")

elif uploaded_files:
    with st.spinner("Wait for it... 🤔"):
        # Process the uploaded files to extract text and source information
        textify_output = read_and_textify(uploaded_files, chunk_size=chunk_size_input)

        # Separate the output into documents (text) and their corresponding sources
        documents, sources = textify_output

        # Call the function
        query_database = list_to_nums(documents)

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Create reference table
            refs_tab = query_search(
                prompt,
                documents,
                query_database,
                sources,
                q_levels,
            )
            refs_tab = refs_tab.head(math.ceil(top_n))
            result = refs_tab

            # Call GPT
            response = call_gpt(prompt, " ".join(list(result.sentences)))

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.write(response)
                with st.expander("See reference:"):
                    st.table(result)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result})
