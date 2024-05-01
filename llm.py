
def generate_response(chunks, idx, prompt):

    """
    return llm response as a str
    """

    input_text = f"""
    You are a helpful AI assistant. Help in answering the following query based on the given contexts.

    Context: {chunks[idx]}

    Query: {prompt}

    """

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

    device = "cpu"
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)
    model = model.to(device)
    outputs = model.generate(**input_ids, max_length=1024)
    print(tokenizer.decode(outputs[0]))

    return tokenizer.decode(outputs[0])

