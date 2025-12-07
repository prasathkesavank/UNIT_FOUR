from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

data_imdb = load_dataset("imdb")
mdl = "distilbert-base-uncased-finetuned-sst-2-english"
sent_pipe = pipeline("sentiment-analysis", model=mdl)

for sample in data_imdb["test"]["text"][:5]:
    print(sent_pipe(sample), "\n")
