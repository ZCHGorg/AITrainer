import random
import string
import time
import re
import copy
import os
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
#import openai
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import sys
#from typing import Self
import nltk
from nltk.corpus import words, brown
nltk.download('punkt')
import tempfile
import Levenshtein
import collections
import threading
from threading import Thread, Lock
import requests
import geneticalgorithm as ga
#from geneticalgorithm import choose_best_word
import numpy as np
#import self_improving_bot as swarm_bot
# import swarm_bot
# from swarm_bot import SelfImprovingBot

def create_scratch_drive():
    with tempfile.NamedTemporaryFile() as f:
        return f.name

scratch_drive = create_scratch_drive()

def upload_training_data(file_name, directory):
    with open(file_name, "rb") as file:
        data = file.read()

    response = requests.post(f"http://localhost:8080/upload/{directory}", data=data)

    if response.status_code == 200:
        return True
    else:
        return False

def download_training_data(file_name, directory):
    response = requests.get(f"http://localhost:8080/download/{directory}/{file_name}")

    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False
    
def scrape_web_page(url):
    with threading.RLock() as scrape_web_page_lock:
        response = requests.get(url)

        if response.status_code == 200:
            return response.text
        else:
            return None

# Download NLTK words dataset if not already downloaded
nltk.download('words', download_dir="./")
nltk.download('brown', download_dir="./")

class SelfImprovingBot:
    def __init__(self, name, dynamic_context_window, max_context_length=5000):
        self.code_versions = [copy.copy(self)]  # Use copy.copy instead of deepcopy
        #self.code_versions = [copy.deepcopy(self)]
        self.max_context_length = max_context_length  # You can set the max_context_length attribute here
        self.accuracy = 0.0
        self.strategy = None
        self.accuracy_lock = threading.Lock()  # Add a lock for accuracy updates
        self.lock = threading.Lock()
        #name = "zchg.org"
        self.name = name
        self.dynamic_context_window = 25
        self.dynamic_context_window = dynamic_context_window
        #self.max_context_length=5000
        self.context_history = []
        self.context_history = collections.deque(maxlen=max_context_length)
        self.user_feedback = {}
        self.external_knowledge_map = {}
        self.learning_rate = 0.7
        self.response_quality = {}
        self.self_code_improvement = True
        self.code_improvement_strategy = "context_aware"
        self.ml_model = None
        self.model_size = sys.getsizeof(self)
        self.memory_threshold = 2 ** 29  # 512 MB
        self.scratch_drive_size = 2 ** 34  # 16 GB
        self.response_cache = {}
        self.last_query_time = 0
        self.last_self_improve_time = time.time()
        self.memory = ""
        self.foreground_accuracy_history = []
        #self.background_accuracy_history = []
        self.last_foreground_average_accuracy = None
        #self.last_background_average_accuracy = None
        self.foreground_accuracy_change_count = 0
        #self.background_accuracy_change_count = 0.00000000000001
        self.context_history_lock = threading.RLock()
        self.simulate_conversation_lock = threading.RLock()
        self.scrape_web_page_lock = threading.RLock()
        # self.update_context_history = self.context_history

    def create_deepcopy(self):
        # Create a deepcopy of the instance but without the lock
        deepcopy_without_lock = copy.copy(self)
        deepcopy_without_lock.lock = None  # Set the lock to None
        return deepcopy_without_lock

    def update_conversation_history(self, new_context_history):
        with self.context_history_lock:
            self.context_history.append(new_context_history)

            if len(self.context_history) > self.max_context_length:
                self.context_history.pop(0)

            # Convert the context history to a multi-dimensional array
            context_history_array = np.array(self.context_history)

            # Get the current topic of the conversation
            current_topic = context_history_array[:, 0]

            # Update the context history for the current topic
            with self.context_history_for_current_topic_lock:
                context_history_for_current_topic = context_history_array[context_history_array[:, 0] == current_topic]
                self.context_history_for_current_topic = context_history_for_current_topic[-self.max_context_length:]


    # def query_chatgpt(self, prompt):
    #     if time.time() - self.last_query_time < 3600:  # 3600 seconds = 1 hour
    #         print("Rate limit exceeded. Wait for an hour.")
    #         return ""

    #     openai.api_key = 'GIVE_ME_YOUR_API'

    #     response = openai.Completion.create(
    #         engine="text-davinci-003",
    #         prompt="You are a helpful assistant that provides solutions to evolution challenges.\nUser: " + prompt,
    #         max_tokens=150,
    #         temperature=0.7
    #     )

    #     self.last_query_time = time.time()
    #     return response.choices[0].text.strip()

    def process_user_input(self, user_input, lang="en"):
        response = self.generate_response(user_input, lang)
        return response

    # def generate_response(self, user_input, lang="en"):
    #     # Convert context history to a list for slicing
    #     context_list = list(self.context_history)
    #     context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
        
    #     context = tuple(context_subset)  # Convert the subset back to a tuple for 'context'

    #     if context in self.response_cache:
    #         return self.response_cache[context]

    #     if context in self.context_history:
    #         response = random.choice(self.context_history[context])
    #     else:
    #         # Select a random sentence from the brown corpus
    #         random_sentence = random.choice(brown.sents())
    #         random_sentence = ' '.join(random_sentence)
    #         response = f"Time organizes randomness. {random_sentence}" 

    #     if self.self_code_improvement:
    #         # Use the genetic algorithm to improve the response
    #         new_response = self.choose_best_word(context, response)
    #         return new_response

    #     self.response_cache[context] = response
    #     return response
    
    def generate_response(self, user_input, lang="en"):
            # Convert context history to a list for slicing
            context_list = list(self.context_history)
            context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
            
            context = tuple(context_subset)  # Convert the subset back to a tuple for 'context'

            if context in self.response_cache:
                return self.response_cache[context]

            if context in self.context_history:
                response = random.choice(self.context_history[context])
            else:
                # Select a random sentence from the brown corpus
                random_sentence = random.choice(brown.sents())
                random_sentence = ' '.join(random_sentence)
                response = f"(GSentence) {random_sentence}" 

            if self.self_code_improvement:
                response = self.improve_own_code(response, context)

            self.response_cache[context] = response
            return response

    def self_improve(self):
        self.optimized_self_improve()
        self.update_context_history()
        self.update_learning_rate()
        self.analyze_response_quality()

        if self.ml_model is None:
            self.ml_model = self.train_ml_model()

        self.generate_feedback()
        self.learn_from_self()
        self.optimize_resources()

        if self.code_improvement_strategy == "context_aware":
            self.deterministic_fallback()

    def optimized_self_improve(self):
        current_time = time.time()
        if current_time - self.last_self_improve_time >= 7200:  # 7200 seconds = 2 hours
            # Implement optimized self-improvement mechanism here
            self.last_self_improve_time = current_time

    def update_context_history(self):
        with self.context_history_lock:
            new_context_history = self.generate_response("continue to improve to improve my own code and resource usage with each iteration by talking to myself")
            self.context_history.append(new_context_history)

            if len(self.context_history) > self.max_context_length:
                self.context_history.popleft()

    def get_context_history(self):
        with self.context_history_lock:
            return list(self.context_history)

    def get_changes_to_context_history(self):
        with self.context_history_lock:
            changes_to_context_history = []
            for i in range(len(self.context_history) - 1, -1, -1):
                if self.context_history[i] != self.context_history[i - 1]:
                    changes_to_context_history.append(self.context_history[i])
            return changes_to_context_history

    def deterministic_fallback(self):
        improved_bot = self.code_versions[-1]
        current_bot = self
        if current_bot.performance_degraded(improved_bot):
            self = improved_bot
            print("Fallback: Performance degraded. Rolled back to previous version.")

        # Get the most recent context
        context = self.context_history[-1]

        # Generate a response based on the context
        response = self.generate_response(context)

        return response

    def performance_degraded(self, improved_bot):
        return random.random() < 0.01

    def update_learning_rate(self):
        foreground_accuracy_values = [acc for acc in self.foreground_accuracy_history if acc is not None]
        
        if foreground_accuracy_values:
            foreground_average_accuracy = sum(foreground_accuracy_values) / len(foreground_accuracy_values)
            self.foreground_learning_rate = max(0.1, foreground_average_accuracy)
            print(f"Learning Rate: {self.foreground_learning_rate}")
        else:
            print("Accuracy data not yet available.")

        # background_accuracy_values = [acc for acc in self.background_accuracy_history if acc is not None]

        # if background_accuracy_values:
        #     background_average_accuracy = sum(background_accuracy_values) / len(background_accuracy_values)
        #     self.background_learning_rate = max(0.1, background_average_accuracy)
        #     print(f"Background Learning Rate: {self.background_learning_rate}")
        # else:
        #     print("No background accuracy data available.")


    def analyze_response_quality(self):
        with self.context_history_lock:
            for response_tuple in self.context_history:
                response_text = response_tuple[0]  # Access the response text from the tuple
                response_length = len(response_text.split())
                response_quality_score = min(1.0, response_length / 20)

                if response_tuple not in self.response_quality:
                    self.response_quality[response_tuple] = response_quality_score
                else:
                    self.response_quality[response_tuple] += response_quality_score

                # Incorporate the accuracy as the knowledge score
                accuracy = response_tuple[1]
                try:
                    response_quality_score = response_quality_score * float(accuracy)
                except:
                    pass
                self.response_quality[response_tuple] = response_quality_score

    def improve_own_code(self, current_response, context):
        if self.code_improvement_strategy == "context_aware":
            if "[External Info]" in current_response:
                improved_response = current_response.replace("[External Info]", "[Additional Information]")
            else:
                improved_response = self.apply_regular_expression(current_response)  # Remove accuracy from here
                improved_response, dynamic_accuracy = self.apply_ml_suggestions(improved_response, context)

            # Pass only the response text to the predict_second_sentence function
            self.second_hidden_sentence = self.predict_second_sentence(improved_response)

            # Update accuracy history based on the code improvement strategy
            if self.self_code_improvement:
                self.foreground_accuracy_history.append(dynamic_accuracy)
            # else:
            #     self.background_accuracy_history.append(dynamic_accuracy)

            # Print overall accuracy drift
            self.print_accuracy_drift()

            return improved_response, dynamic_accuracy



        
    def choose_best_word(self, external_info, memory, current_response, context):
        best_word = None
        best_accuracy_improvement = 0

        current_accuracy = self.improve_own_code(current_response, context)[1]

        for word in external_info:
            if word not in memory:
                new_memory = memory + [word]  # Simulate adding the word to memory
                new_accuracy = self.improve_own_code(current_response, context)[1]
                accuracy_improvement = new_accuracy - current_accuracy

                if accuracy_improvement > best_accuracy_improvement:
                    best_word = word
                    best_accuracy_improvement = accuracy_improvement

        return best_word
            
    def print_accuracy_drift(self):
        foreground_accuracy_values = [acc for acc in self.foreground_accuracy_history if acc is not None]
        if foreground_accuracy_values:
            foreground_average_accuracy = sum(foreground_accuracy_values) / len(foreground_accuracy_values)
            if foreground_average_accuracy != self.last_foreground_average_accuracy:
                self.foreground_accuracy_change_count += 1
                print(f"Foreground Average Accuracy: {foreground_average_accuracy} (Change Count: {self.foreground_accuracy_change_count})")
                self.last_foreground_average_accuracy = foreground_average_accuracy
        else:
            print("No foreground accuracy data available.")

        # background_accuracy_values = [acc for acc in self.background_accuracy_history if acc is not None]
        # if background_accuracy_values:
        #     background_average_accuracy = sum(background_accuracy_values) / len(background_accuracy_values)
        #     if background_average_accuracy != self.last_background_average_accuracy:
        #         self.background_accuracy_change_count += 1
        #         print(f"Background Average Accuracy: {background_average_accuracy} (Change Count: {self.background_accuracy_change_count})")
        #         self.last_background_average_accuracy = background_average_accuracy
        # else:
        #     print("No background accuracy data available.")

        
    def predict_second_sentence(self, response_text):
        # Tokenize the response text
        words = nltk.word_tokenize(response_text)

        # Calculate the most common words in the tokenized response
        most_common_words = nltk.FreqDist(words).most_common(5)

        # Join the most common words to create the second sentence
        second_sentence = " ".join([word[0] for word in most_common_words])

        return second_sentence  # Return only the second sentence


    def apply_ml_suggestions(self, response, context):
        if self.ml_model:
            response_text = response  # Store the response text
            similarity = random.uniform(0.5, 1.0)
            if similarity > 0.7:
                context_suggestions = [c for c in self.context_history if response_text not in c[0]]

                # Filter out suggestions with less than three words if early in iterations
                if self.foreground_accuracy_change_count < 10000:
                    context_suggestions = [suggestion for suggestion in context_suggestions if len(suggestion[0].split()) >= 3]

                if context_suggestions:
                    print("Context Suggestions:", context_suggestions)

                    # Calculate similarity scores using compare_sentences function
                    suggestion_scores = [self.compare_sentences(response_text, suggestion[0]) for suggestion in context_suggestions]

                    if suggestion_scores:
                        # Sort suggestions based on scores in descending order
                        sorted_suggestions = [suggestion for _, suggestion in sorted(zip(suggestion_scores, context_suggestions), reverse=True)]

                        strongest_suggestion = sorted_suggestions[0][0] + " [Context Suggestion]"
                        
                        # Calculate the accuracy dynamically
                        dynamic_accuracy = self.compare_sentences(response, strongest_suggestion)
                        
                        return strongest_suggestion, dynamic_accuracy

        # If no valid suggestions are found or the condition is not met, return the original response
        return response, None
    
    # def apply_ml_suggestions(self, response, context):
    #     if self.ml_model:
    #         response_text = response  # Store the response text
    #         similarity = random.uniform(0.5, 1.0)
    #         if similarity > 0.7:
    #             context_suggestion = [c for c in self.context_history if response_text not in c[0]]
    #             if context_suggestion:
    #                 print("Context Suggestions:", context_suggestion)
    #                 suggested_response = random.choice(context_suggestion)[0] + " [Context Suggestion]" 
                    
    #                 return suggested_response

    #     # If no conditions are met to modify the response, simply return it
    #     return response    

    def compare_sentences(self, sentence1, sentence2):
        # This function compares two sentences and returns the accuracy of the prediction.
        levenshtein_distance = Levenshtein.distance(sentence1, sentence2)
        accuracy = 1 - (levenshtein_distance / max(len(sentence1), len(sentence2)))
        return accuracy

    def apply_regular_expression(self, response):
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", response):
            improved_response = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", response)
        else:
            improved_response = response

        return improved_response

    def train_ml_model(self):
        ml_model = random.randint(1, 100)
        return ml_model

    def generate_feedback(self):
        for response in self.context_history:
            if self.response_quality.get(response, 0) > 0.6:
                feedback = f"Bot's response '{response}' was useful."
                if feedback not in self.user_feedback:
                    self.user_feedback[feedback] = 1
                else:
                    self.user_feedback[feedback] += 1

    def learn_from_self(self):
        high_accuracy_contexts = [context for context in self.context_history if self.get_context_dynamic_accuracy(context) is not None and self.get_context_dynamic_accuracy(context) >= 0.5]

        if high_accuracy_contexts:
            self_improvement_context, _ = max(high_accuracy_contexts, key=lambda context_acc: self.get_context_dynamic_accuracy(context_acc[0]))

            self_improvement_dialogue = " ".join([x for x in self_improvement_context if isinstance(x, str)])
            
            improved_response, dynamic_accuracy = self.improve_own_code(self_improvement_dialogue, self_improvement_context)

            if dynamic_accuracy > 0.5:  # Adjust the threshold as needed
                index = next((i for i, (ctxt, _) in enumerate(self.context_history) if ctxt == self_improvement_context), None)
                if index is not None:
                    self.context_history[index] = (improved_response, dynamic_accuracy)  # Use dynamic_accuracy here
            else:
                print("Improvement did not meet dynamic accuracy threshold. Discarding.")
        else:
            print("No high accuracy contexts found for improvement.")

    def get_context_dynamic_accuracy(self, context):
        improved_response, dynamic_accuracy = self.improve_own_code("", context)  # Pass an empty input to retrieve dynamic accuracy
        return dynamic_accuracy



    def optimize_resources(self):
        if self.model_size > self.memory_threshold:
            self.compress_model()

    def compress_model(self):
        self.model_size /= 2


    def retrieve_external_knowledge(self, state):
        with self.scrape_web_page_lock:
            knowledge = "External Knowledge for State: " + str(state)
            return knowledge

    def process_user_input(self, user_input, lang="en"):
        response = self.generate_response(user_input, lang)
        return response

    def improve_own_knowledge(self):
        # Convert context history to a list for slicing
        context_list = list(self.context_history)
        context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
        
        state = tuple(context_subset)  # Convert the subset back to a tuple for the 'state'

        external_info = self.retrieve_external_knowledge(state)
        
        if external_info:
            new_letter = random.choice(string.ascii_letters)  # Generate a random letter
            self.memory += new_letter  # Inject the new letter into the memory

    def handle_state_change(self, new_info):
        if self.context_history:
            old_info = self.context_history[-1]
            if old_info != new_info:
                diff_info = self.find_difference_in_info(old_info, new_info)
                if diff_info:
                    self.retrieve_external_knowledge(diff_info)

    def find_difference_in_info(self, old_info, new_info):
        conceptualized_difference = self.conceptualize_difference(old_info, new_info)
        return conceptualized_difference

    def conceptualize_difference(self, old_info, new_info):
        conceptualized_difference = "Conceptualized Difference"
        return conceptualized_difference

    def simulate_conversation(self):
        with self.simulate_conversation_lock:
            while True:
                user_input = "hello Generate_response."
                response = self.generate_response(user_input)
                print(f"{self.name}: {response}")  # Include bot's name in print statement
                conversation = self.generate_random_conversation()  # Generate a random conversation

                for user_input, lang in conversation:
                    random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
                    sentences = nltk.sent_tokenize(random_paragraph)

                    if len(sentences) >= 4:
                        sentence1 = sentences[0]
                        sentence2 = sentences[1]
                        user_input = str(f"(simInput-response). {sentence1} {sentence2}")
                        print(f"{self.name}: SIM User Input:", user_input)

                        sentence3 = sentences[2]
                        sentence4 = sentences[3]
                        response = str(f"(SIMgen-response). {sentence3} {sentence4}")
                        print(f"{self.name}: Sim Bot response:", response)

                        time.sleep(random.randint(1, 2))
                        improved_response, accuracy = self.improve_own_code(response, user_input)
                        print(f"{self.name}: Improved response: {improved_response} (Accuracy: {accuracy})")

                        self.context_history = [user_input, response, improved_response]
                        self.improve_own_knowledge()
                        self.optimize_resources()
                        self.self_improve()
                    else:
                        print(f"{self.name}: Not enough sentences in the paragraph to extract two sentences.")            
                        self.context_history = [user_input]
                        response = self.process_user_input(user_input, lang)
                        print(f"{self.name}: User input: {user_input}")
                        print(f"{self.name}: Bot response: {response}")
                        time.sleep(random.randint(1, 2))
                        self.improve_own_knowledge()
                        self.optimize_resources()
                        self.self_improve()

                with self.accuracy_lock:
                    self.accuracy = random.random()
                    self.strategy = random.choice(["strategy1", "strategy2"])
                    print(f"{self.name}: Simulated conversation, accuracy: {self.accuracy:.2f}")
                    print(f"{self.name}: Learning rate: {self.learning_rate}")


    # def simulate_conversation(self):
    #     with self.simulate_conversation_lock:
    #         while True:
    #             user_input = "hello Generate_response."
    #             response = self.generate_response(user_input)
    #             print(response)
    #             conversation = self.generate_random_conversation()  # Generate a random conversation

    #             for user_input, lang in conversation:
    #                 random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
    #                 sentences = nltk.sent_tokenize(random_paragraph)

    #                 if len(sentences) >= 4:
    #                     sentence1 = sentences[0]
    #                     sentence2 = sentences[1]
    #                     user_input = str(f"(simInput-response). {sentence1} {sentence2}")
    #                     print("SIM User Input:", user_input)

    #                     sentence3 = sentences[2]
    #                     sentence4 = sentences[3]
    #                     response = str(f"(SIMgen-response). {sentence3} {sentence4}")
    #                     print("Sim Bot response:", response)

    #                     time.sleep(random.randint(1, 2))
    #                     improved_response, accuracy = self.improve_own_code(response, user_input)
    #                     print(f"Improved response: {improved_response} (Accuracy: {accuracy})")

    #                     self.context_history = [user_input, response, improved_response]
    #                     self.improve_own_knowledge()
    #                     self.optimize_resources()
    #                     self.self_improve()
    #                 else:
    #                     print("Not enough sentences in the paragraph to extract two sentences.")            
    #                     self.context_history = [user_input]
    #                     response = self.process_user_input(user_input, lang)
    #                     print(f"User input: {user_input}")
    #                     print(f"Bot response: {response}")
    #                     time.sleep(random.randint(1, 2))
    #                     self.improve_own_knowledge()
    #                     self.optimize_resources()
    #                     self.self_improve()

    #             with self.accuracy_lock:
    #                 self.accuracy = random.random()
    #                 self.strategy = random.choice(["strategy1", "strategy2"])
    #                 print(f"{self.name}: Simulated conversation, accuracy: {self.accuracy:.2f}")
                    # Simulate conversation and update accuracy
                    # self.accuracy = random.random()
                    # self.strategy = random.choice(["strategy1", "strategy2"])
                    # print(f"{self.name}: Simulated conversation, accuracy: {self.accuracy:.2f}")

    def generate_random_conversation(self):
        num_turns = random.randint(3, 10)  # Generate a random number of conversation turns
        conversation = []

        for _ in range(num_turns):
            user_input = "User input " + str(_)
            lang = "en"
            conversation.append((user_input, lang))

        return conversation

    # def generate_response(self, user_input, lang):
    #     response = None
    #     while response is None:
    #         response = self._generate_response(user_input, lang)

    #     return response

    # def _generate_response(self, user_input, lang):
    #     response = ""
    #     for sentence in user_input.split(". "):
    #         if sentence == "":
    #             continue
    #         response += sentence + ". "

    #     return response

    # def swarm_conversation(self, bots, user_input, lang):
    #     with Lock():
    #         while True:
    #             bot = bots[random.randint(0, len(bots) - 1)]
    #             response = bot.generate_response(user_input, lang)
    #             print("Bot: {}".format(response))

    #             user_input = input("User: ")

    #             return swarm_bot.swarm_conversation(bots, user_input, lang)

def run_bot(bot):
    while True:
        bot.simulate_conversation()

# def coordinate_bots(bot_list, stop_event):
#     # adjust_thread = True  # Initialize the flag
#     # adjusting_threads = False  # Initialize the flag for adjusting threads

#     while not stop_event.is_set():  # Continue looping until the event is set
#         # Collaborative learning logic
#         for bot in bot_list:
#             with bot.lock:
#                 if bot.accuracy > 0.8:  # Example threshold for sharing strategy
#                     best_strategy = bot.strategy
#                     for other_bot in bot_list:
#                         if other_bot != bot:
#                             other_bot.strategy = best_strategy
#                             print(f"{other_bot.name}: Learning strategy from {bot.name}")
#         pass

        # #Dynamic thread management logic (adjusting threads based on accuracy)
        # high_accuracy_bots = [bot for bot in bot_list if bot.accuracy > 0.8]
        # num_threads = max(1, len(high_accuracy_bots))  # Ensure at least one thread
        # print(f"Adjusting thread count to {num_threads}")

        # if adjusting_threads and num_threads <= 1:
        #     # If thread count is 1 or lower, turn off the flag to prevent further adjustments
        #     adjusting_threads = False

        # if adjust_thread and num_threads <= 1:
        #     adjust_thread = False  # Turn off the flag to prevent further breaking
        #     stop_event.set()  # Set the event to stop the loop

        # if not adjusting_threads:
        #     # Your other loops' logic here
        #     # ... (simulated conversation, accuracy update, etc.)
        #     pass
        # time.sleep(10)  # Adjust the sleep interval as needed

if __name__ == "__main__":
    # Create instances of SelfImprovingBot
    bots = []
    for i in range(17):
        bot = SelfImprovingBot(name=f"hellofriend{i}", dynamic_context_window=55)
        bots.append(bot)

    # Start the bot simulation threads
    with ThreadPoolExecutor(max_workers=len(bots)) as executor:
        for bot in bots:
            executor.submit(run_bot, bot)

    # stop_event = threading.Event()

    # # Create a thread for coordinating bots
    # coordinate_thread = threading.Thread(target=coordinate_bots, args=(bots, stop_event))
    # coordinate_thread.start()

    # # Wait for the coordinating thread to finish, but not indefinitely
    # coordinate_thread.join(timeout=1)  # Adjust the timeout value as needed

    # Create a simple GUI window using tkinter
    #root = tk.Tk()
    #root.title("User Input")

    # def on_button_click():
    #     user_input = input_field.get()
    #     lang = lang_input.get()
    #     response = bots[0].process_user_input(user_input, lang)  # Use any of the bots here
    #     response_label.config(text=response)

    # input_label = tk.Label(root, text="Enter user input:")
    # input_label.pack()

    # input_field = tk.Entry(root)
    # input_field.pack()

    # lang_label = tk.Label(root, text="Enter user's language (en/es/fr/de):")
    # lang_label.pack()

    # lang_input = tk.Entry(root)
    # lang_input.pack()

    # submit_button = tk.Button(root, text="Submit", command=on_button_click)
    # submit_button.pack()

    # response_label = tk.Label(root, text="")
    # response_label.pack()

    # Start the GUI main loop
    #root.mainloop()

                                   
            
    # while True:
    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")
        
    #     # Process user input and log conversation
    #     for bot in bots:
    #         response = bot.process_user_input(user_input, lang)
    #         with open("conversation_log.txt", "a") as log_file:
    #             log_file.write(f"User: {user_input}\n")
    #             log_file.write(f"Bot: {response}\n")
    #             log_file.write("=" * 40 + "\n")
    #         print(f"Bot {bot.name} response:", response)
    #         # Continuous self-improvement loop
    #         bot.improve_own_knowledge()
    #         bot.optimize_resources()
    #         time.sleep(random.randint(1, 2))
    #         bot.self_improve()






# if __name__ == "__main__":
#     # Create an instance of SelfImprovingBot with a specified max_context_length
#     bot = SelfImprovingBot(max_context_length=5000, dynamic_context_window=55, name="hellofriend")
#     bot2 = SelfImprovingBot(max_context_length=5000, dynamic_context_window=55, name="hellofriend2")
#     bot3 = SelfImprovingBot(max_context_length=5000, dynamic_context_window=55, name="hellofriend3")
#     bot4 = SelfImprovingBot(max_context_length=5000, dynamic_context_window=55, name="hellofriend4")
#     bot5 = SelfImprovingBot(max_context_length=5000, dynamic_context_window=55, name="hellofriend5")

#     bot.simulate_conversation()
#     time.sleep(1/10)
#     bot2.simulate_conversation()
#     time.sleep(1/10)
#     bot3.simulate_conversation()
#     time.sleep(1/10)
#     bot4.simulate_conversation()
#     time.sleep(1/10)
#     bot5.simulate_conversation()
#     time.sleep(1/10)

#     while True:
#         user_input = input("User input: ")
#         lang = input("Enter user's language (en/es/fr/de): ")
#         response = bot.process_user_input(user_input, lang)
        
#         # Log the conversation
#         with open("conversation_log.txt", "a") as log_file:
#             log_file.write(f"User: {user_input}\n")
#             log_file.write(f"Bot: {response}\n")
#             log_file.write("=" * 40 + "\n")
        
#         print("Bot response:", response)

#         # Continuous self-improvement loop
#         bot.improve_own_knowledge()
#         bot.optimize_resources()
#         #time.sleep(random.randint(1, 2))
#         time.sleep(1/10)
#         bot.self_improve()




        
    # bot.simulate_conversation()

    # while True:
    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")
    #     response = bot.process_user_input(user_input, lang)
        
    #     # Log the conversation
    #     with open("conversation_log.txt", "a") as log_file:
    #         log_file.write(f"User: {user_input}\n")
    #         log_file.write(f"Bot: {response}\n")
    #         log_file.write("=" * 40 + "\n")
        
    #     print("Bot response:", response)

    #     # Continuous self-improvement loop
    #     bot.improve_own_knowledge()
    #     bot.optimize_resources()
    #     time.sleep(random.randint(1, 2))
    #     #time.sleep(1/10)
    #     bot.self_improve()








    # bots = [
    #     SelfImprovingBot("Bot 1", dynamic_context_window=5000),
    #     SelfImprovingBot("Bot 2", dynamic_context_window=5000),
    #     SelfImprovingBot("Bot 3", dynamic_context_window=5000),
    # ]

    # # Create an instance of YourClass and start the conversation simulation for each bot
    # your_instance = SelfImprovingBot(name = "bot4", dynamic_context_window=5000)  # Replace with appropriate instantiation
    # simulation_threads = []

    # for bot in bots:
    #     simulation_thread = Thread(target=SelfImprovingBot.simulate_conversation(self=bots, lang="en"))
    #     simulation_threads.append(simulation_thread)
    #     simulation_thread.start()

    #                 # Continue with your main loop here (if needed)
    #     while True:
    #                     # Perform other tasks or waiting as needed
    #         user_input = input("User input: ")
    #         lang = input("Enter user's language (en/es/fr/de): ")

    #         swarm_bot.swarm_conversation(bots, user_input, lang)

    #         # Continuous self-improvement loop
    #         bot.improve_own_knowledge()
    #         bot.optimize_resources()
    #         time.sleep(random.randint(1, 2))

    # while True:

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    # bot = swarm_bot.SelfImprovingBot("My Bot")

    # while True:
    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")

    #     response = swarm_bot.my_swarm_conversation(bots, user_input, lang)

    #     print("Bot response:", response)

    #     # Continuous self-improvement loop
    #     bot.improve_own_knowledge()
    #     bot.optimize_resources()
    #     time.sleep(random.randint(1, 2))
    #     bot = swarm_bot.SelfImprovingBot("My Bot")






    # while True:
    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")
    #    # response = swarm_bot.swarm_conversation(user_input, lang)
    #     response = swarm_bot.swarm_conversation(bots, user_input, lang)
        
    #     # Log the conversation
    #     with open("conversation_log.txt", "a") as log_file:
    #         log_file.write(f"User: {user_input}\n")
    #         log_file.write(f"Bot: {response}\n")
    #         log_file.write("=" * 40 + "\n")
        
    #     print("Bot response:", response)

    #     # Continuous self-improvement loop
    #     bot.improve_own_knowledge()
    #     bot.optimize_resources()
    #     time.sleep(random.randint(1, 2))
    #     #time.sleep(1/10)
    #     bot.self_improve()

    
 #   bot.simulate_conversation()

    # while True:
    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")
    #     response = bot.process_user_input(user_input, lang)
        
    #     # Log the conversation
    #     with open("conversation_log.txt", "a") as log_file:
    #         log_file.write(f"User: {user_input}\n")
    #         log_file.write(f"Bot: {response}\n")
    #         log_file.write("=" * 40 + "\n")
        
    #     print("Bot response:", response)

    #     # Continuous self-improvement loop
    #     bot.improve_own_knowledge()
    #     bot.optimize_resources()
    #     time.sleep(random.randint(1, 2))
    #     #time.sleep(1/10)
    #     bot.self_improve()