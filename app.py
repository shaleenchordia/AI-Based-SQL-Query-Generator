import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./models/trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI
st.title("AI-Based SQL Query Generator")
st.write("Enter your natural language query below, and the AI will generate the corresponding SQL query.")

user_input = st.text_area("Enter your query:", "How many employees earn more than $50,000?")

if st.button("Generate SQL"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=150)
        sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
        
        st.subheader("Generated SQL Query:")
        st.code(sql_query, language="sql")
    else:
        st.warning("Please enter a valid input.")
