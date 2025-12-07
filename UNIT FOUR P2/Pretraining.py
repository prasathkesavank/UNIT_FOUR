# Using distilgpt for textgeneration

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Chat generation using mistralai/Mistral-7B-Instruct-v0.2

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"   
)

chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = chat_pipeline("Hello! Explain how black holes form.", max_length=150)
print(response[0]['generated_text'])


# Summarize text using T5-small

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

text = """
Machine learning is a method of data analysis that automates analytical model building. 
It allows systems to learn from data and make decisions with minimal human intervention.
"""
input_text = "summarize: " + text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=2, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
