# -*- coding: utf-8 -*-
"""vector embeddings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1waGbRGbGraqeL9VRpjEeDuv4Ap0yPjuy
"""

import numpy as np
import torch
from torch import nn

corpus = [
    "The sky is blue.",
    "The sun is bright today.",
    "She enjoys reading books on rainy days.",
    "He loves playing soccer with his friends.",
    "They traveled to the mountains last summer.",
    "She cooks delicious meals every weekend.",
    "The flowers in the garden are blooming beautifully.",
    "He drives his car to work every day.",
    "She is learning to play the piano.",
    "They are planning a trip to the beach.",
    "The cat sleeps on the couch all afternoon.",
    "He reads the newspaper in the morning.",
    "She is studying for her final exams.",
    "They are building a new house in the countryside.",
    "The coffee shop on the corner makes great lattes.",
    "He likes to go jogging in the park.",
    "She is painting a portrait of her dog.",
    "They went to the zoo last weekend.",
    "The bakery sells fresh bread every morning.",
    "He enjoys listening to jazz music.",
    "She works as a teacher at the local school.",
    "They adopted a puppy from the animal shelter.",
    "The children are playing in the backyard.",
    "He likes to watch movies on Friday nights.",
    "She loves hiking in the forest during the fall.",
    "They visited the art museum in the city.",
    "The river flows gently through the valley.",
    "He practices yoga every morning before work.",
    "She is attending a photography workshop next week.",
    "They are hosting a barbecue party this Saturday.",
    "The bookstore downtown has a great selection of novels.",
    "He is learning to cook Italian cuisine.",
    "She enjoys writing poetry in her free time.",
    "They are planning to renovate their kitchen next month.",
    "The rain falls softly on the roof.",
    "He watches the sunset from his balcony.",
    "She is knitting a scarf for her friend.",
    "They went camping in the national park.",
    "The chef at the restaurant prepares exquisite dishes.",
    "He plays the guitar in a local band.",
    "She enjoys swimming in the ocean during the summer.",
    "They are planting trees in the community garden.",
    "The wind blows gently through the open window.",
    "He collects vintage records from the 1980s.",
    "She is learning a new language in her spare time.",
    "They are organizing a charity event for the homeless.",
    "The dog runs across the field chasing a ball.",
    "He reads historical novels during his lunch break.",
    "She loves watching the stars on clear nights.",
    "They are training for a marathon together.",
    "The birds sing in the trees every morning.",
    "He is taking a photography class at the art center.",
    "She practices playing the violin in the evenings.",
    "They are restoring an old farmhouse in the countryside.",
    "The movie theater downtown shows classic films.",
    "He likes to sketch landscapes in his notebook.",
    "She is learning to sew her own clothes.",
    "They adopted a kitten from the animal rescue.",
    "The leaves on the trees are turning golden in autumn.",
    "He is writing a novel set in the 19th century.",
    "She enjoys baking cookies for her neighbors.",
    "They are planning to visit the museum of modern art.",
    "The sun sets behind the mountains, painting the sky.",
    "He collects rare stamps from around the world.",
    "She is designing a website for a local business.",
    "They went on a road trip across the country.",
    "The rain is pouring down heavily this evening.",
    "He enjoys fishing in the lake near his house.",
    "She is decorating her apartment with new furniture.",
    "They are volunteering at the animal shelter this weekend.",
    "The snow is falling softly outside the window.",
    "He is studying marine biology at the university.",
    "She likes to draw portraits of people in her sketchbook.",
    "They went skiing in the mountains last winter.",
    "The library has a large collection of historical books.",
    "He practices speaking French with his classmates.",
    "She is learning to play chess with her brother.",
    "They are remodeling their living room this summer.",
    "The farmer harvests corn in the field every autumn.",
    "He enjoys cycling through the city on weekends.",
    "She is preparing a presentation for her class.",
    "They are attending a music festival next month.",
    "The waves crash against the shore at the beach.",
    "He enjoys birdwatching in the forest behind his house.",
    "She is writing a research paper on environmental science.",
    "They went horseback riding in the countryside.",
    "The bakery is known for its chocolate croissants.",
    "He is taking a course on digital marketing.",
    "She likes to run along the river in the mornings.",
    "They are visiting a botanical garden this afternoon.",
    "The flowers in the park are blooming in the springtime.",
    "He collects old coins from different countries.",
    "She is learning to play the drums in a music class.",
    "They are hosting a dinner party for their friends.",
    "The artist is painting a mural on the side of the building.",
    "He enjoys reading science fiction novels at night.",
    "She is making handmade jewelry for an art show.",
    "They are planning to travel to Europe next year.",
    "The children are drawing pictures with colorful crayons.",
    "He practices archery at the local sports club.",
    "She is learning to code in Python for her job."
]

# import re

# def preprocess(corpus: list):
#   _corpus = [re.sub(r'[.,]', '', sentence) for sentence in corpus]
#   return [sentence.lower() for sentence in _corpus]



# new_corpus = preprocess(corpus)

import re

class Preprocess:

  def __init__(self, corpus: list):
    self.corpus = corpus

  def preprocess(self):
    _corpus = [re.sub(r'[.,]', '', sentence) for sentence in self.corpus]
    return [sentence.lower() for sentence in _corpus]


class Tokenize:

  def __init__(self, corpus: list):
    self.corpus = corpus

  def tokenize(self):
    tokenized_ = []
    for sentence in self.corpus:
      tokenized_.append(sentence.split())
    return tokenized_

new_corpus = Preprocess(corpus).preprocess()
tokenized_corpus = Tokenize(new_corpus).tokenize()
print(tokenized_corpus)

from collections import defaultdict


def build_vocab(tokenized_corpus):
    word2idx = {}
    idx2word = {}
    idx = 0
    for sentence in tokenized_corpus:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    return word2idx, idx2word

word2idx, idx2word = build_vocab(tokenized_corpus)
print("Word to index mapping:", word2idx)
print("Index to word mapping:", idx2word)

def generate_word_context_pairs(tokenized_corpus, window_size=1):
    word_context_pairs = []
    for sentence in tokenized_corpus:
        for i, word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            context_words = [sentence[j] for j in range(start, end) if j != i]
            for context in context_words:
                word_context_pairs.append((word, context))
    return word_context_pairs

pairs = generate_word_context_pairs(tokenized_corpus, window_size=1)
print(pairs)

def word_pairs_to_indices(pairs, word2idx):
    indices = [(word2idx[word], word2idx[context]) for word, context in pairs]
    return indices

pair_indices = word_pairs_to_indices(pairs, word2idx)
print(pair_indices)

import numpy as np

def initialize_embeddings(vocab_size, embedding_dim):
    return np.random.randn(vocab_size, embedding_dim)

embedding_dim = 5  # We're using 2 dimensions for simplicity
vocab_size = len(word2idx)
embeddings = initialize_embeddings(vocab_size, embedding_dim)
print(embeddings)

import torch
import torch.nn.functional as F

def train_simple_skipgram(pairs, embeddings, learning_rate=1, epochs=100):
    embeddings = torch.tensor(embeddings, requires_grad=True)

    for epoch in range(epochs):
        total_loss = 0
        for target, context in pairs:
            # Compute dot product
            target_vector = embeddings[target]
            context_vector = embeddings[context]
            score = torch.dot(target_vector, context_vector)

            # Loss (negative log likelihood)
            loss = -F.logsigmoid(score)

            # Backpropagate
            loss.backward()

            # Update embeddings with gradient descent
            with torch.no_grad():
                embeddings[target] -= learning_rate * embeddings.grad[target]
                embeddings[context] -= learning_rate * embeddings.grad[context]

            embeddings.grad.zero_()  # Clear gradients
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {total_loss}')

    return embeddings.detach().numpy()

trained_embeddings = train_simple_skipgram(pair_indices, embeddings, learning_rate=1, epochs=200)

import matplotlib.pyplot as plt

def plot_embeddings(embeddings, word2idx, idx2word):
    for word, idx in word2idx.items():
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1])
        plt.text(embeddings[idx, 0], embeddings[idx, 1], word)
    plt.show()

plot_embeddings(trained_embeddings, word2idx, idx2word)

import numpy as np

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to find top N most similar words to a target word
def find_similar_words(target_word, word2idx, embeddings, top_n=3):
    # Get the embedding for the target word
    target_index = word2idx[target_word]
    target_embedding = embeddings[target_index]

    # Calculate similarity of target word with all other words in the vocabulary
    similarities = {}
    for word, index in word2idx.items():
        if word != target_word:  # Don't compare the word to itself
            word_embedding = embeddings[index]
            similarity = cosine_similarity(target_embedding, word_embedding)
            similarities[word] = similarity

    # Sort the words by similarity score in descending order
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Return the top N most similar words
    return similar_words[:top_n]

# Example: Find words closest to 'sky'
similar_words_to_sky = find_similar_words('is', word2idx, embeddings)

# Print the most similar words
print("Words closest to 'sky':")
for word, similarity in similar_words_to_sky:
    print(f"{word}: {similarity:.4f}")

