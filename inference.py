from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "./models/trained_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_sql(query):
    inputs = tokenizer(query, return_tensors="pt")
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_query = input("Enter your natural language query: ")
    sql_query = generate_sql(user_query)
    print("Generated SQL Query:", sql_query)
