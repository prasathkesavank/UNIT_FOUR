import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = [
    "facebook/bart-large-cnn",
    "t5-small",
    "google/pegasus-xsum"
]

geo_text = """
The Himalayan mountain range spans five countries and is home to the highest peaks on Earth, including Mount Everest. 
The Amazon rainforest, often called the lungs of the planet, covers large parts of South America and hosts diverse wildlife. 
Coastal regions are increasingly affected by rising sea levels due to climate change. 
Deserts like the Sahara are known for extreme temperatures and sparse vegetation. 
Rivers such as the Nile, Amazon, and Yangtze play crucial roles in the ecology and human civilizations surrounding them.
"""

for model_name in models_list:
    print(f"\n=========== Model: {model_name} ===========\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    inputs = tokenizer.encode(geo_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(
        inputs,
        max_length=80,
        num_beams=4,
        length_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)
