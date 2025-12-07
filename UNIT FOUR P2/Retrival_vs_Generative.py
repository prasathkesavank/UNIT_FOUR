from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

qa_pairs = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there!",
    "what is ai": "AI is the simulation of human intelligence by machines.",
    "what is chatbot": "A chatbot is a software that interacts with users through text.",
    "bye": "Goodbye! Have a nice day."
}

vectorizer = TfidfVectorizer()
questions = list(qa_pairs.keys())
tfidf_matrix = vectorizer.fit_transform(questions)

def retrieval_bot(user_input):
    user_vec = vectorizer.transform([user_input])
    idx = cosine_similarity(user_vec, tfidf_matrix).argmax()
    return qa_pairs[questions[idx]]

def generative_bot(user_input):
    templates = [
        f"That's interesting! You said: '{user_input}'",
        f"I'm not sure about that, but let's think: '{user_input}'",
        f"Here’s something related to what you asked: '{user_input}'",
        f"Wow! You mentioned '{user_input}'. Let's explore that."
    ]
    return random.choice(templates)

def compare_responses(retrieval_reply, generative_reply):
    if retrieval_reply == generative_reply:
        style = "Both responses are similar."
    elif retrieval_reply in generative_reply or generative_reply in retrieval_reply:
        style = "Responses overlap slightly."
    else:
        style = "Responses are different: Retrieval is predictable, Generative is creative."
    return style

print("Chatbot comparison (type 'exit' or 'bye' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "bye"]:
        print("Bot: Goodbye!")
        break

    retrieval_reply = retrieval_bot(user_input)
    generative_reply = generative_bot(user_input)
    comparison = compare_responses(retrieval_reply, generative_reply)

    print("Retrieval Bot:", retrieval_reply)
    print("Generative Bot:", generative_reply)
    print("Comparison:", comparison)
    print("-" * 50)
