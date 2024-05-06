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

        import json, time

        start_time = time.time()

        from llama_cpp import Llama

        llm = Llama(
            model_path="./models\llama-2-7b.Q4_0.gguf",
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
        )
        response = llm(
            f"""
            Q: You are a helpful AI assistant. Help in answering the following query based on the given contexts.

            Context: {chunks[idx]}

            Query: {prompt}

            A: """, # Prompt
            max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        # print(output)

        # response = requests.post(url=url, data=json.dumps(data)).text

        st.info(f"time taken for response {time.time() - start_time}")

        with st.chat_message("assistant"):

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})