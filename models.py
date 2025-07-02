import os
import numpy as np

from SparseVec import SparseVector
from gensim.models import Word2Vec


class EmbeddingModel:
	"""
	Base class responsible for handling deck and card embeddings.
	"""
	def get_card_embedding(self, card_id):
		"""
		Virtual method: Computes and returns the embedding corresponding to a card.
		"""
		pass

	def get_deck_embedding(self, card_list):
		"""
		Virtual method: Computes and returns the embedding corresponding to a deck.
		"""
		pass

	def dot(self, a, b):
		"""
		Virtual method: Computes the dot-product of two embedding vectors and returns it.
		"""
		pass


class Model:
	"""
	Base class for converting card ids to vectors.
	"""

	def mean_vector(self, card_list):
		"""
		Virtual method: Returns a vector representing a set of cards.
		"""
		pass

	def get_vector(self, card_id):
		"""
		Virtual method: Returns the vector associated with a certain card id.
		"""
		pass

	def get_idf_score(self, card_id):
		"""
		Virtual method: Returns the idf score of a card.
		"""

	def get_dot_function(self):
		return self.dot

	@staticmethod
	def dot(a, b):
		"""
		Virtual method: Computes the dot product of two vectors and returns them.
		"""
		pass


class CFEmbedding(EmbeddingModel):
	def __init__(self, index):
		self.index = index

	def get_card_embedding(self, card_id):
		return self.index[card_id]

	def get_deck_embedding(self, card_list):
		deck_vector = SparseVector()
		valid_cards = 0
		for card_id in card_list:
			if card_id not in self.index:
				continue
			deck_vector = deck_vector + self.get_card_embedding(card_id)
			valid_cards += 1
		deck_vector = deck_vector / valid_cards
		return deck_vector

	def dot(self, a, b):
		return SparseVector.dot(a, b)

class Card2VecEmbedding(EmbeddingModel):
	pass


class Suggester:
	pass


class SuggestKNN(Suggester):
	pass


class CFModel(Model):
	"""
	Model using SparseVectors and an inverted index as the embedding vectors.
	"""
	def __init__(self, index):
		self.index = index

	def mean_vector(self, card_list):
		deck_vector = SparseVector()
		for card_id in card_list:
			if card_id not in self.index:
				continue
			deck_vector = deck_vector + self.index[card_id]
		deck_vector = deck_vector / len(card_list)
		return deck_vector

	def get_vector(self, card_id):
		return self.index[card_id]

	def get_idf_score(self, card_id):
		return self.index.idf_scores[card_id]

	@staticmethod
	def dot(a, b):
		"""
		Computes the dot product of a and b and returns it.
		"""
		return SparseVector.dot(a, b)


class ProbabilityModel(Model):
	# Needs card_id_to_decks index.
	pass


class Card2VecModel(Model):
	"""
	Model using SparseVectors and an inverted index as the embedding vectors.
	"""
	def __init__(self, corpus, index, model_path="models/card2vec.model"):
		# TODO: Needs index too (or idf score at least).
		self.corpus = corpus
		self.index = index
		if os.path.exists(model_path):
			self.model = self._load_model(model_path)
		else:
			self.model = self._train_model(model_path)

	def _load_model(self, model_path):
		return Word2Vec.load(model_path).wv

	def _train_model(self, model_path):
		vector_dimension = 100
		window = 90
		model = Word2Vec(sentences=self.corpus, vector_size=vector_dimension, window=window, min_count=0)
		model.save(model_path)
		return model.wv

	def mean_vector(self, card_list):
		return np.array(self.model.get_mean_vector(card_list))

	def get_vector(self, card_id):
		return self.model[card_id]

	def get_idf_score(self, card_id):
		return self.index.idf_scores[card_id]

	def get_dot_function(self):
		return self.dot

	@staticmethod
	def dot(a, b):
		"""
		Computes the dot product of a and b and returns it.
		"""
		return np.dot(a, b)

