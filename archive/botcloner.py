import random
import string
import time
import re
import copy
#import openai
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import sys
import nltk
from nltk.corpus import words, brown
nltk.download('punkt')
import tempfile
import Levenshtein
import collections
import threading
from threading import Lock
import requests
import geneticalgorithm as ga
#from geneticalgorithm import choose_best_word
import numpy as np
#import self_improving_bot as swarm_bot
# import swarm_bot
#from swarm_bot import SelfImprovingBot, swarm_conversation

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

#Download NLTK words dataset if not already downloaded
nltk.download('words', download_dir="./")
nltk.download('brown', download_dir="./")

class self:
    def __init__(self, name, dynamic_context_window, max_context_length=5000):
        self.max_context_length = max_context_length  # You can set the max_context_length attribute here
        name = "zchg.org"
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
        self.code_versions = [copy.deepcopy(self)]
        self.last_query_time = 0
        self.last_self_improve_time = time.time()
        self.memory = ""
        self.foreground_accuracy_history = []
        self.background_accuracy_history = []
        self.last_foreground_average_accuracy = None
        self.last_background_average_accuracy = None
        self.foreground_accuracy_change_count = 0
        self.background_accuracy_change_count = 0
        self.context_history_lock = threading.RLock()
        self.simulate_conversation_lock = threading.RLock()
        self.scrape_web_page_lock = threading.RLock()
        # self.update_context_history = self.context_history


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
            response = f"Time organizes randomness. {random_sentence}" 


        if self.self_code_improvement:
            # Use the genetic algorithm to improve the response
            new_response = self.choose_best_word(context, response)
            return new_response

        self.response_cache[context] = response
        return response
    
    # def generate_response(self, user_input, lang="en"):
    #         # Convert context history to a list for slicing
    #         context_list = list(self.context_history)
    #         context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list
            
    #         context = tuple(context_subset)  # Convert the subset back to a tuple for 'context'

    #         if context in self.response_cache:
    #             return self.response_cache[context]

    #         if context in self.context_history:
    #             response = random.choice(self.context_history[context])
    #         else:
    #             # Select a random sentence from the brown corpus
    #             random_sentence = random.choice(brown.sents())
    #             random_sentence = ' '.join(random_sentence)
    #             response = f"Time organizes randomness. {random_sentence}" 

    #         if self.self_code_improvement:
    #             response = self.improve_own_code(response, context)

    #         self.response_cache[context] = response
    #         return response

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
        # Simulate the previous conversation
        with self.simulate_conversation_lock:
            self.simulate_conversation(self.context_history[-1])

        # Get the most recent context
        context = self.context_history[-1]

        # Generate a response based on the context
        response = self.generate_response(context)

        return response

    def performance_degraded(self, improved_bot):
        return random.random() < 0.01

    def update_learning_rate(self):
        if self.foreground_accuracy_history:
            foreground_average_accuracy = sum(self.foreground_accuracy_history) / len(self.foreground_accuracy_history)
            if foreground_average_accuracy > self.last_foreground_average_accuracy:
                self.learning_rate = min(0.7, self.learning_rate * 1.1)
            elif foreground_average_accuracy < self.last_foreground_average_accuracy:
                self.learning_rate = max(0.1, self.learning_rate / 1.1)

            self.last_foreground_average_accuracy = foreground_average_accuracy
        else:
            print("No foreground accuracy data available.")

        if self.background_accuracy_history:
            background_average_accuracy = sum(self.background_accuracy_history) / len(self.background_accuracy_history)
            if background_average_accuracy > self.last_background_average_accuracy:
                self.learning_rate = min(0.7, self.learning_rate * 1.1)
            elif background_average_accuracy < self.last_background_average_accuracy:
                self.learning_rate = max(0.1, self.learning_rate / 1.1)

            self.last_background_average_accuracy = background_average_accuracy
        else:
            print("No background accuracy data available.")

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
                improved_response = self.apply_ml_suggestions(improved_response, context)

            # Pass only the response text to the predict_second_sentence function
            second_hidden_sentence = self.predict_second_sentence(improved_response)
            accuracy = self.compare_sentences(improved_response, second_hidden_sentence)

            # Update accuracy history based on the code improvement strategy
            if self.self_code_improvement:
                self.foreground_accuracy_history.append(accuracy)
            else:
                self.background_accuracy_history.append(accuracy)

            # Print overall accuracy drift
            self.print_accuracy_drift()

            return improved_response, accuracy
        
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
        if self.foreground_accuracy_history:
            foreground_average_accuracy = sum(self.foreground_accuracy_history) / len(self.foreground_accuracy_history)
            if foreground_average_accuracy != self.last_foreground_average_accuracy:
                self.foreground_accuracy_change_count += 1
                print(f"Foreground Average Accuracy: {foreground_average_accuracy} (Change Count: {self.foreground_accuracy_change_count})")
                self.last_foreground_average_accuracy = foreground_average_accuracy
        else:
            print("No foreground accuracy data available.")

        if self.background_accuracy_history:
            background_average_accuracy = sum(self.background_accuracy_history) / len(self.background_accuracy_history)
            if background_average_accuracy != self.last_background_average_accuracy:
                self.background_accuracy_change_count += 1
                print(f"Background Average Accuracy: {background_average_accuracy} (Change Count: {self.background_accuracy_change_count})")
                self.last_background_average_accuracy = background_average_accuracy
        else:
            print("No background accuracy data available.")
        
    def predict_second_sentence(self, response_text):
        # Tokenize the response text
        words = nltk.word_tokenize(response_text)

        # Calculate the most common words in the tokenized response
        most_common_words = nltk.FreqDist(words).most_common(5)

        # Join the most common words to create the second sentence
        second_sentence = " ".join([word[0] for word in most_common_words])

        return second_sentence

    def apply_ml_suggestions(self, response, context):
        if self.ml_model:
            response_text = response  # Store the response text
            similarity = random.uniform(0.5, 1.0)
            if similarity > 0.7:
                context_suggestions = [c for c in self.context_history if response_text not in c[0]]
                if context_suggestions:
                    print("Context Suggestions:", context_suggestions)

                    # Calculate similarity scores using compare_sentences function
                    suggestion_scores = [self.compare_sentences(response_text, suggestion[0]) for suggestion in context_suggestions]

                    # Sort suggestions based on scores in descending order
                    sorted_suggestions = [suggestion for _, suggestion in sorted(zip(suggestion_scores, context_suggestions), reverse=True)]

                    strongest_suggestion = sorted_suggestions[0][0] + " [Context Suggestion]"
                    return strongest_suggestion

        # If no conditions are met to modify the response, simply return it
        return response
    
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
        self_improvement_context = random.choice(self.context_history)
        self_improvement_dialogue = " ".join([x for x in self_improvement_context if isinstance(x, str)])
        improved_code, accuracy = self.improve_own_code(self_improvement_dialogue, self_improvement_context)
        self.context_history[self.context_history.index(self_improvement_context)] = improved_code


    def optimize_resources(self):
        if self.model_size > self.memory_threshold:
            self.compress_model()

    def compress_model(self):
        self.model_size /= 2


    def retrieve_external_knowledge(self, state):
        with self.scrape_web_page_lock:
            knowledge = "External Knowledge for State: " + str(state)
            return knowledge

    def improve_own_knowledge(self):
        # Convert context history to a list for slicing
        context_list = list(self.context_history)
        context_subset = context_list[-self.dynamic_context_window:]  # Slicing on a list

        state = tuple(context_subset)  # Convert the subset back to a tuple for the 'state'

        external_info = self.retrieve_external_knowledge(state)

        if external_info:
            if self.foreground_accuracy_change_count % 1000 == 0:
                # Use the scrape_web_page function as the external source
                with self.scrape_web_page_lock:
                    external_info = scrape_web_page("https://en.wikipedia.org/wiki/Synonym")

                # Use the NLTK library to generate synonyms
                for word in external_info:
                    for synonym in nltk.wordnet.synsets(word):
                        if synonym not in self.memory:
                            self.memory += synonym.name()

            # Use a genetic algorithm to choose the best word to inject into the memory
            best_word = self.choose_best_word(external_info, self.memory)
            with self.scrape_web_page_lock:
                self.memory += best_word  # Inject the new word into the memory

            print(f"Improved knowledge: {self.memory}")

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
    #                     user_input = str(f"Time organizes randomness(simInput-response). {sentence1} {sentence2}")
    #                     print("SIM User Input:", user_input)

    #                     sentence3 = sentences[2]
    #                     sentence4 = sentences[3]
    #                     response = str(f"Time organizes randomness(SIMgen-response). {sentence3} {sentence4}")
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
    # def simulate_conversation(self):
    #         with self.simulate_conversation_lock:
    #             while True:
    #                 user_input = "hello Generate_response."
    #                 response = self.generate_response(user_input)
    #                 print(response)
    #                 conversation = self.generate_random_conversation()  # Generate a random conversation

    #                 for user_input, lang in conversation:
    #                     random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
    #                     sentences = nltk.sent_tokenize(random_paragraph)

    #                     if len(sentences) >= 4:
    #                         sentence1 = sentences[0]
    #                         sentence2 = sentences[1]
    #                         user_input = str(f"Time organizes randomness(simInput-response). {sentence1} {sentence2}")
    #                         print("SIM User Input:", user_input)

    #                         sentence3 = sentences[2]
    #                         sentence4 = sentences[3]
    #                         response = str(f"Time organizes randomness(SIMgen-response). {sentence3} {sentence4}")
    #                         print("Sim Bot response:", response)

    #                         time.sleep(random.randint(1, 2))
    #                         improved_response, accuracy = self.improve_own_code(response, user_input)
    #                         print(f"Improved response: {improved_response} (Accuracy: {accuracy})")

    #                         self.context_history = [user_input, response, improved_response]
    #                         # Create a new SelfImprovingBot object
    #                         bot = swarm_bot.SelfImprovingBot("My Bot", 10)
    #                         # Call the swarm_conversation() method on the SelfImprovingBot object
    #                         response = swarm_bot.swarm_conversation(bot, user_input, lang)
    #                         print("Bots: {}".format(response))
    #                     else:
    #                         print("Not enough sentences in the paragraph to extract two sentences.")            
    #                         self.context_history = [user_input]
    #                         response = self.process_user_input(user_input, lang)
    #                         print(f"User input: {user_input}")
    #                         print(f"Bot response: {response}")
    #                         time.sleep(random.randint(1, 2))
    #                         self.improve_own_knowledge()
    #                         self.optimize_resources()
    #                         self.self_improve()



    def simulate_conversation(self):
            with self.simulate_conversation_lock:
                while True:
                    user_input = "hello Generate_response."
                    response = self.generate_response(user_input)
                    print(response)
                    conversation = self.generate_random_conversation()  # Generate a random conversation

                    for user_input, lang in conversation:
                        random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))
                        sentences = nltk.sent_tokenize(random_paragraph)

                        if len(sentences) >= 4:
                            sentence1 = sentences[0]
                            sentence2 = sentences[1]
                            user_input = str(f"Time organizes randomness(simInput-response). {sentence1} {sentence2}")
                            print("SIM User Input:", user_input)

                            sentence3 = sentences[2]
                            sentence4 = sentences[3]
                            response = str(f"Time organizes randomness(SIMgen-response). {sentence3} {sentence4}")
                            print("Sim Bot response:", response)

                            time.sleep(random.randint(1, 2))
                            improved_response, accuracy = self.improve_own_code(response, user_input)
                            print(f"Improved response: {improved_response} (Accuracy: {accuracy})")

                            self.context_history = [user_input, response, improved_response]
                            self.improve_own_knowledge()
                            self.optimize_resources()
                            self.self_improve()
                        else:
                            print("Not enough sentences in the paragraph to extract two sentences.")            
                            self.context_history = [user_input]
                            bots = [bot for bot in bots if bot.is_alive()]
                            if len(bots) > 0:
                                bot = bots[random.randint(0, len(bots) - 1)]
                                response = bot.simulate_conversation(user_input, lang)
                            else:
                                response = self.process_user_input(user_input, lang)
                            print(f"User input: {user_input}")
                            print(f"Bot response: {response}")
                            time.sleep(random.randint(1, 2))
                            self.improve_own_knowledge()
                            self.optimize_resources()
                            self.self_improve()

    def generate_random_conversation(self):
        num_turns = random.randint(3, 10)  # Generate a random number of conversation turns
        conversation = []

        for _ in range(num_turns):
            user_input = "User input " + str(_)
            lang = "en"
            conversation.append((user_input, lang))

        return conversation

    def _generate_response(self, user_input, lang):
        response = ""
        for sentence in user_input.split(". "):
            if sentence == "":
                continue
            response += sentence + ". "

        return response

    def swarm_conversation(self, bots, user_input, lang):
        with Lock():
            while True:
                bot = bots[random.randint(0, len(bots) - 1)]
                response = bot.generate_response(user_input, lang)
                print("Bot: {}".format(response))

                user_input = input("User: ")

                return swarm_bot.swarm_conversation(bots, user_input, lang)

    # if __name__ == "__main__":
    #     bots = [
    #     swarm_bot.create_bot("Bot 1", dynamic_context_window=5000),
    #     swarm_bot.create_bot("Bot 2", dynamic_context_window=5000),
    #     swarm_bot.create_bot("Bot 3", dynamic_context_window=5000),
    #     ]

    #     bot = swarm_bot.create_bot("My Bot", dynamic_context_window=5000)

    #     while True:
    #         bot.simulate_conversation()

    #         user_input = input("User input: ")
    #         lang = input("Enter user's language (en/es/fr/de): ")

    #         swarm_bot.swarm_conversation(bots, user_input, lang)

    #         for bot in bots:
    #             if bot.is_alive:
    #                 bot.improve_own_knowledge()
    #                 bot.optimize_resources()

    #         print("No user input for 10 seconds. Exiting.")
    #         break






    # bots = [
    #     SelfImprovingBot("Bot 1", dynamic_context_window=5000),
    #     SelfImprovingBot("Bot 2", dynamic_context_window=5000),
    #     SelfImprovingBot("Bot 3", dynamic_context_window=5000),
    # ]

    # bot = SelfImprovingBot("My Bot", dynamic_context_window=5000)

    # while True:
    #     bot.simulate_conversation()

    #     user_input = input("User input: ")
    #     lang = input("Enter user's language (en/es/fr/de): ")

    #     # Start a timer to generate responses every few seconds
    #     start_time = time.time()
    #     elapsed_time = 0
    #     while elapsed_time < 1000:  # Generate responses for 10 seconds
    #         swarm_conversation(bots, user_input, lang)

    #         # Continuous self-improvement loop
    #         bot.improve_own_knowledge()
    #         bot.optimize_resources()

    #         elapsed_time = time.time() - start_time
    #         time.sleep(random.randint(1, 2))  # Add a delay between responses

    #     print("No user input for 10 seconds. Exiting.")
    #     break  # Exit the main loop after 10 seconds without input
            
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    
    #     bot.simulate_conversation()

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
    #         time.sleep(random.randint(1, 2))
    #         #time.sleep(1/10)
    #         bot.self_improve()