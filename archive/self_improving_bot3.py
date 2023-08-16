import random
import string
import time
import re
import copy
# import openai
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import sys
import nltk
from nltk.corpus import words, brown
import tempfile
import Levenshtein
import threading
import itertools

def create_scratch_drive():
    with tempfile.NamedTemporaryFile() as f:
        return f.name

scratch_drive = create_scratch_drive()
# Download NLTK words dataset if not already downloaded
nltk.download('words', download_dir="./")
nltk.download('brown', download_dir="./")
nltk.download('punkt')

class SelfImprovingBot:
    def __init__(self):
        self.user_feedback = {}
        self.context_history = []
        self.external_knowledge_map = {}
        self.max_context_length = 5
        self.dynamic_context_window = 2
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
        self.lang = "en"
        self.response_cache = {}
        super().__init__()
        self.self_learn = True


#    def query_chatgpt(self, prompt):
#        if time.time() - self.last_query_time < 3600:  # 3600 seconds = 1 hour
#            print("Rate limit exceeded. Wait for an hour.")
#            return ""
#
#        openai.api_key = 'GIVE_ME_YOUR_API'

#        response = openai.Completion.create(
#            engine="text-davinci-003",
#            prompt="You are a helpful assistant that provides solutions to evolution challenges.\nUser: " + prompt,
#            max_tokens=150,
#            temperature=0.7
#        )

#        self.last_query_time = time.time()
#        return response.choices[0].text.strip()

    def generate_response(self, user_input, lang="en"):
        context = tuple(self.context_history[-self.dynamic_context_window:])

        if context in self.response_cache:
            return self.response_cache[context]

        if context in self.context_history:
            response = random.choice(self.context_history[context])
        else:
            # Select a random sentence from the brown corpus
            # random_sentence = random.choice(brown.sents())
            # random_sentence = ' '.join(random_sentence)
            # response = str(f"Time organizes randomness(gen-response). {random_sentence}")

            # Generate a random paragraph (a group of sentences) from the corpus
            random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))

            # Tokenize the paragraph into sentences
            sentences = nltk.sent_tokenize(random_paragraph)

            # Ensure there are at least two sentences
            if len(sentences) >= 2:
                # Extract two consecutive sentences from the same paragraph
                sentence1 = sentences[0]
                sentence2 = sentences[1]
                response = str(f"Time organizes randomness(gen-response). {sentence1} {sentence2}")
                print("Generated response:", response)
            else:
                print("Not enough sentences in the paragraph to extract two sentences.")

        if self.self_code_improvement:
            response = self.improve_own_code(response, context)

        self.response_cache[context] = response
        return response
    
    def _self_learn(self):
            # Perform self-learning tasks
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

    def self_improve(self):
            # Run self-learning in a separate thread
            self_learning_thread = threading.Thread(target=self._self_learn)
            self_learning_thread.daemon = True
            self_learning_thread.start()

            # Prompt the bot with new inputs
            # random_sentence = random.choice(brown.sents())
            # random_sentence = ' '.join(random_sentence)
            # user_input = str(f"Time organizes randomness(self-improve). {random_sentence}")

            # Generate a random paragraph (a group of sentences) from the corpus
            random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))

            # Tokenize the paragraph into sentences
            sentences = nltk.sent_tokenize(random_paragraph)

            # Ensure there are at least two sentences
            if len(sentences) >= 2:
                # Extract two consecutive sentences from the same paragraph
                sentence1 = sentences[0]
                sentence2 = sentences[1]
                user_input = str(f"Time organizes randomness(gen-response). {sentence1} {sentence2}")
                print("Generated response:", user_input)
            else:
                print("Not enough sentences in the paragraph to extract two sentences.")            

            # Get the user's language preference
            lang = "en"

            # Process the user input
            #response = self.process_user_input(user_input, lang)
            
           

            # Print the response
            #print("Bot response2:", response)

            # Perform self-improvement tasks
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
        self.context_history.append(self.generate_response("continue to improve to improve my own code and resource usage with each iteration by talking to myself"))

        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)

    def deterministic_fallback(self):
        improved_bot = self.code_versions[-1]
        current_bot = self
        if current_bot.performance_degraded(improved_bot):
            self = improved_bot
            print("Fallback: Performance degraded. Rolled back to previous version.")

    def performance_degraded(self, improved_bot):
        return random.random() < 0.01

    def update_learning_rate(self):
        self.learning_rate = min(0.7, 0.1 + len(self.context_history) / 1000)

    def analyze_response_quality(self):
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

            return improved_response, accuracy
        
        elif self.code_improvement_strategy == "ml_based":
            improved_response = self.apply_ml_suggestions(current_response, context)

            # Pass only the response text to the predict_second_sentence function
            second_hidden_sentence = self.predict_second_sentence(improved_response)
            accuracy = self.compare_sentences(improved_response, second_hidden_sentence)

            return improved_response, accuracy
        else:
            return current_response, 0.0  # Return the original response and a default accuracy

    def predict_second_sentence(self, response_text):
        # Tokenize the response text
        words = nltk.word_tokenize(response_text)

        # Calculate the most common words in the tokenized response
        most_common_words = nltk.FreqDist(words).most_common(5)

        # Join the most common words to create the second sentence
        second_sentence = " ".join([word[0] for word in most_common_words])

        return second_sentence

    def compare_sentences(self, sentence1, sentence2):
        # This function compares two sentences and returns the accuracy of the prediction.

        levenshtein_distance = Levenshtein.distance(sentence1, sentence2)
        accuracy = 1 - (levenshtein_distance / len(sentence1))

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
    
    def apply_ml_suggestions(self, response, context):
        if self.ml_model:
            response_text = response  # Store the response text
            similarity = random.uniform(0.5, 1.0)
            if similarity > 0.7:
                context_suggestion = [c for c in self.context_history if response_text not in c[0]]
                if context_suggestion:
                    suggested_response = random.choice(context_suggestion)[0] + " [Context Suggestion]"
                    return suggested_response

        # If no conditions are met to modify the response, simply return it
        return response
        
    def get_context_suggestion(self, context):
            context_suggestion = []
            for c in self.context_history:
                if context not in c and c not in context_suggestion:
                    context_suggestion.append(c)
            return context_suggestion

    def generate_feedback(self):
        for response in self.context_history:
            if self.response_quality.get(response, 0) > 0.6:
                feedback = f"Bot's response '{response}' was useful."
                print("useful!")
                if feedback not in self.user_feedback:
                    self.user_feedback[feedback] = 1
                    print("+1 FEEDBACK!")
                else:
                    self.user_feedback[feedback] += 1
                    print("NOT +1 FEEDBACK!")

    def learn_from_self(self):
        self_improvement_context = random.choice(self.context_history)
        self_improvement_dialogue = " ".join([x for x in self_improvement_context if type(x) == str])
        improved_code, accuracy_score = self.improve_own_code(self_improvement_dialogue, self_improvement_context)
        self.context_history[self.context_history.index(self_improvement_context)] = improved_code

    def optimize_resources(self):
        if self.model_size > self.memory_threshold:
            self.compress_model()

    def compress_model(self):
        self.model_size /= 2
        self.self_improve()

    def improve_own_knowledge(self):
        state = tuple(self.context_history[-self.dynamic_context_window:])
        external_info = self.retrieve_external_knowledge(state)
        
        if external_info:
            new_info = external_info
            self.memory = new_info
#            print("new knowledge, oh boy!")
            self.self_improve()
        else:
#            print("no external source")
            self.self_improve()
            new_letter = random.choice(string.ascii_letters)  # Generate a random letter
            self.memory += new_letter  # Inject the new letter into the memory

    def simulate_conversation(self):
        random_sentence = random.choice(brown.sents())
        random_sentence = ' '.join(random_sentence)
        initial_user_input = str(f"{random_sentence}")
        conversation = self.generate_random_conversation(initial_user_input)  # Generate a random conversation
        for user_input, lang in conversation:
            #response = self.process_user_input(user_input, lang)
            #random_sentence = random.choice(brown.sents())
            #random_sentence = ' '.join(random_sentence)
            #user_input = str(f"Time organizes randomness(simulate conversation). {random_sentence}")
            #print(f"User input: {user_input}")
            # Generate a random paragraph (a group of sentences) from the corpus
            random_paragraph = " ".join(" ".join(sentence) for sentence in (random.choice(brown.sents()) for _ in range(5)))

            # Tokenize the paragraph into sentences
            sentences = nltk.sent_tokenize(random_paragraph)

            # Ensure there are at least four sentences
        if len(sentences) >= 4:
                # Extract two consecutive sentences from the same paragraph
            sentence1 = sentences[0]
            sentence2 = sentences[1]
            user_input = str(f"Time organizes randomness(SIMgen-response). {sentence1} {sentence2}")
            print("User Input:", user_input)


            # random_sentence2 = random.choice(brown.sents())
            # random_sentence2 = ' '.join(random_sentence2)  # Corrected this line       
            # response = str({random_sentence2})  # Corrected this line
            # print(f"Bot response: {response}")
            
            # Generate a random paragraph (a group of sentences) from the corpus
            #random_paragraph2 = " ".join(random.choice(brown.sents()) for _ in range(5))  # Adjust the number of sentences as needed

            # Tokenize the paragraph into sentences
            # sentences2 = nltk.sent_tokenize(random_paragraph2)

            # Ensure there are at least two sentences
            # if len(sentences2) >= 2:
                # Extract two consecutive sentences from the same paragraph
            sentence3 = sentences[2]
            sentence4 = sentences[3]
            response = str(f"Time organizes randomness(SIM2gen-response). {sentence3} {sentence4}")
            print("Bot response:", response)

            # time.sleep(random.randint(1, 6))
            improved_response, accuracy = self.improve_own_code(response, user_input)  # Unpack tuple and pass response only
            #improved_response, accuracy = self.improve_own_code(response[0], user_input)
            print(f"Improved response: {improved_response} (Accuracy: {accuracy})")
            self.improve_own_knowledge()
            self.optimize_resources()
            self.self_improve()
            self.simulate_conversation()
        else:
            print("Not enough sentences in the paragraph to extract two sentences.")

    def generate_random_conversation(self, initial_user_input):
        num_turns = random.randint(3, 10)  # Generate a random number of conversation turns
        conversation = [(initial_user_input, "en")]

        for _ in range(num_turns):
            user_input = "User input " + str(_)
            lang = "en"
            conversation.append((user_input, lang))

        return conversation
            
    def retrieve_external_knowledge(self, state):
        print("retrive_external_knowledge")
        knowledge = "External Knowledge for State: " + str(state)
        return knowledge

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

if __name__ == "__main__":
    bot = SelfImprovingBot()
    bot.simulate_conversation()

    while True:
        random_sentence = random.choice(brown.sents())
        random_sentence = ' '.join(random_sentence)
        user_input = str(f"Time organizes randomness(whitetrue). {random_sentence}")
        lang = "en"
        response = str({random_sentence})
        #response = bot.process_user_input(user_input, lang)
        
        # Log the conversation
        with open("conversation_log.txt", "a") as log_file:
            log_file.write(f"User: {user_input}\n")
            log_file.write(f"Bot: {response}\n")
            log_file.write("=" * 40 + "\n")
        
 #       print("Bot response:", response)

        # Continuous self-improvement loop
        bot.improve_own_knowledge()
        bot.optimize_resources()
        time.sleep(random.randint(1, 2))
        bot.self_improve()