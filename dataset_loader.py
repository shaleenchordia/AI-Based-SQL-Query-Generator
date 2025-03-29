from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("motherduckdb/duckdb-text2sql-25k")
    return dataset["train"]

def format_example(example):
    return f"Query: {example['query']}\nSQL: {example['sql']}"
