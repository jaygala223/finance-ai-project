import streamlit as st
from pathlib import Path
import os
import requests
import chunk_adapter, embedder, utils, llm, reader

st.title("Finance Assistant")

sidebar = st.sidebar

if "file_parsed" not in st.session_state:
        st.session_state.file_parsed = False

with sidebar:
    
    uploaded_file = st.file_uploader("Please upload your financial documents", type=['pdf'])

    if uploaded_file:
        save_folder = 'uploaded_files/'
        save_path = os.path.join(save_folder, uploaded_file.name)

        print(f"SAVE PATH: {save_path}")
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        if os.path.exists(save_path):
            st.success(f'File {uploaded_file.name} is successfully saved!')

        st.session_state.file_parsed = True

if st.session_state.file_parsed:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter prompt"):

        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        text = reader.extract_text(save_path)

        chunks = chunk_adapter.chunk_text(text)

        chunk_embeddings = embedder.create_chunk_embeddings(chunks)

        prompt_embeddings = embedder.create_prompt_embeddings(prompt)

        idx, _, _ = utils.find_closest_chunk(prompt_embedding=prompt_embeddings, chunk_embeddings=chunk_embeddings)

        url = "http://localhost:8000/generate_response"

        # print("TYPE: ", type(chunks))

        # print("DATA: ", data)

        data={
            "chunks": chunks,
            "idx": int(idx),
            "prompt": prompt
        }

        import json

        response = requests.post(url=url, data=json.dumps(data)).text

        with st.chat_message("assistant"):

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})