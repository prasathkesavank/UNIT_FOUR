!pip install sumy
import nltk
nltk.download('punkt')

from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

text = """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.
Specific applications of AI include expert systems, natural language processing, speech recognition, and machine vision.
AI research has been going on since the 1950s and has made significant progress in recent years due to advances in computing power and data availability.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
textrank = TextRankSummarizer()
extractive = textrank(parser.document, sentences_count=2)
extractive_text = " ".join([str(s) for s in extractive])

abstractive_model = pipeline("summarization")
abstractive_text = abstractive_model(text, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]

print("\nExtractive Summary:\n", extractive_text)
print("\nAbstractive Summary:\n", abstractive_text)

print("\n--- Comparison ---")
print("Extractive Summary Style: Uses original sentences from the text.")
print("Abstractive Summary Style: Generates new sentences with the same meaning.")
print("Extractive Word Count:", len(extractive_text.split()))
print("Abstractive Word Count:", len(abstractive_text.split()))

common = set(extractive_text.lower().split()) & set(abstractive_text.lower().split())
print("Common Words in Both Summaries:", len(common))

print("\nConclusion:")
print("TextRank performs extractive summarization by selecting important sentences.")
print("HuggingFace performs abstractive summarization by rewriting content in a natural, shorter form.")
!pip install sumy
import nltk
nltk.download('punkt')

from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

text = """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.
Specific applications of AI include expert systems, natural language processing, speech recognition, and machine vision.
AI research has been going on since the 1950s and has made significant progress in recent years due to advances in computing power and data availability.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
textrank = TextRankSummarizer()
extractive = textrank(parser.document, sentences_count=2)
extractive_text = " ".join([str(s) for s in extractive])

abstractive_model = pipeline("summarization")
abstractive_text = abstractive_model(text, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]

print("\nExtractive Summary:\n", extractive_text)
print("\nAbstractive Summary:\n", abstractive_text)

print("\n--- Comparison ---")
print("Extractive Summary Style: Uses original sentences from the text.")
print("Abstractive Summary Style: Generates new sentences with the same meaning.")
print("Extractive Word Count:", len(extractive_text.split()))
print("Abstractive Word Count:", len(abstractive_text.split()))

common = set(extractive_text.lower().split()) & set(abstractive_text.lower().split())
print("Common Words in Both Summaries:", len(common))

print("\nConclusion:")
print("TextRank performs extractive summarization by selecting important sentences.")
print("HuggingFace performs abstractive summarization by rewriting content in a natural, shorter form.")
