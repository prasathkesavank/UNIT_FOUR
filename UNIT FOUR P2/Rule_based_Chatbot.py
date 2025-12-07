import random

answers = {
    "hi": ["Hey! How’s it going?", "Hello! How can I help you today?"],
    "hello": ["Hi there!", "Hey! What’s up?"],
    "what is ai": ["AI is when machines try to think like humans."],
    "what is chatbot": ["A chatbot is a program that talks to you."],
    "who made python": ["Python was created by Guido van Rossum."],
    "thank you": ["No worries!", "You’re welcome!"],
    "bye": ["See you!", "Bye! Take care!"]
}

def reply(user_text):
    user_text = user_text.lower()
    for word in answers:
        if word in user_text:
            return random.choice(answers[word])
    return "Hmm, I’m not sure about that. Can you ask something else?"

print("Bot: Hi! Ask me anything. Type 'bye' to exit.")
while True:
    user_text = input("You: ")
    if user_text.lower() == "bye":
        print("Bot:", random.choice(answers["bye"]))
        break
    print("Bot:", reply(user_text))

