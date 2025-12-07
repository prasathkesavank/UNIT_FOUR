!pip install transformers accelerate sentencepiece

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    device_map="auto",
    max_new_tokens=200
)

#zero-shot prompting
prompt = "Explain what a microcontroller is."

output = pipe(prompt)[0]["generated_text"]
print(output)

#one-shot prompting
prompt = """
Q: What is a resistor?
A: A resistor restricts current flow.

Q: What is a microcontroller?
A:"""

!pip install transformers accelerate sentencepiece pandas

from transformers import pipeline
import pandas as pd

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    device_map="auto",
    max_new_tokens=200
)

prompts = {
    "Zero-shot": "Explain what climate change is in simple words.",
    
    "One-shot": """
Q: What is the Great Wall of China?
A: It is a series of fortifications built to protect China from invasions.

Q: What is the Amazon Rainforest?
A:""",
    
    "Few-shot": """
You are a geography teacher.

Q: What is Mount Everest?
A: It is the highest mountain on Earth, located in the Himalayas.

Q: What is the Sahara Desert?
A: It is the largest hot desert in the world, located in North Africa.

Q: What is the Nile River?
A: It is the longest river in the world, flowing through northeastern Africa.

Q: What is the Pacific Ocean?
A:""",
    
    "Chain-of-thought": "Explain how the greenhouse effect works step by step, including how gases trap heat in the Earth's atmosphere.",
    
    "Role-based": "You are an environmental scientist. Explain global warming in a simple way for students.",
    
    "Context-based": """
Using the context below, explain climate change in three simple sentences.

Context:
The Earth is warming because of increased greenhouse gases from human activities such as burning fossil fuels.
This causes melting glaciers, rising sea levels, and extreme weather events.

Answer:
"""
}

results = {}

for key, prompt in prompts.items():
    output = generator(prompt)[0]["generated_text"]
    results[key] = output.strip()

df = pd.DataFrame(list(results.items()), columns=["Prompting Technique", "Output"])
pd.set_option("display.max_colwidth", 500)
print(df)

