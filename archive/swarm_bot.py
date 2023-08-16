import random
from threading import Lock
import nltk

def swarm_conversation(bots, user_input, lang):
    with Lock():
        while True:
            for bot in bots:
                response = bot.generate_response(user_input, lang)
                print("{}: {}".format(bot.name, response))

            user_input = input("User: ")

            if user_input.lower() == 'exit':
                break

class SelfImprovingBot:
    def __init__(self, name):
        self.name = name  # Store the name as an instance variable

    def generate_response(self, user_input, lang):
        response = ""
        for n in range(1, 5):
            response += self.generate_n_gram(user_input, lang, n)
        return response

    def generate_n_gram(self, user_input, lang, n):
        tokens = nltk.word_tokenize(user_input)
        ngrams = nltk.ngrams(tokens, n)
        response = ""
        for ngram in ngrams:
            response += " ".join(ngram) + " "
        return response
