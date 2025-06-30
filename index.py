import glob
import math
import random
import datasets
import utils
from SparseVec import SparseVector


class InvertedIndex:
	def __init__(self, dataset_path):
		# TODO: If dataset does not exist, generate it.

		self.inverted_index = {}
		self.idf_scores = {}
		self.create_index(dataset_path)

	def add_cards_to_index(self, i, cards):
		for card_id in cards:
			card_id = card_id.lstrip("0")
			if card_id not in self.inverted_index:
				self.inverted_index[card_id] = SparseVector()
			self.inverted_index[card_id][i] = 1

	def compute_idf_scores(self, number_of_decks):
		for card_id in self.inverted_index:
			self.idf_scores[card_id] = math.log(number_of_decks / len(self.inverted_index[card_id]))

	def normalise_inverted_index(self):
		for card_id in self.inverted_index:
			vector = self.inverted_index[card_id]
			self.inverted_index[card_id] = vector / vector.norm

	def create_index(self, dataset_path):
		dataset = datasets.load_from_disk(dataset_path)
		for i, sample in enumerate(dataset):
			if (i + 1) % 1000 == 0:
				print("Progress indexing files:", i + 1, "files indexed.")

			self.add_cards_to_index(i, sample["cards"])

		self.compute_idf_scores(len(dataset))
		self.normalise_inverted_index()

	def __iter__(self):
		for x in self.inverted_index.keys():
			yield x

	def __getitem__(self, item):
		return self.inverted_index[item]


class Corpus:
	def __init__(self, dataset_path):
		"""
		Creates a corpus from an iterable containing decks/lists of cards, used to train card2vec models.
		"""
		self.corpus = []
		dataset = datasets.load_from_disk(dataset_path)
		for sample in dataset:
			card_ids = sample["cards"]
			random.shuffle(card_ids)
			self.corpus.append(card_ids)

	def __iter__(self):
		for line in self.corpus:
			yield line


def generate_dataset(directory, dataset_name):
	list_of_decks = []
	for i, file_name in enumerate(glob.glob(f"{directory}*.ydk", recursive=True)):
		if (i + 1) % 1000 == 0:
			print(f"Currently creating dataset for directory: {directory}.", i + 1, "files indexed.")
		cards = utils.load_deck(file_name)
		if len(cards) == 0:
			continue
		sample = {
			"filename": file_name,
			"cards": cards
		}
		list_of_decks.append(sample)

	dataset = datasets.Dataset.from_list(list_of_decks)
	dataset.save_to_disk(f"Datasets/card_lists/{dataset_name}")


"""def index_decks(dataset_path):
	number_of_decks = 0
	card_index = 0
	dataset = datasets.load_from_disk(dataset_path)
	for i, sample in dataset:
		if (i+1) % 1000 == 0:
			print("Progress indexing files:", i+1, "files indexed.")

		cards = sample["cards"]

		deck_to_card_ids[i] = cards
		for card_id in cards:
			if card_id not in card_id_to_decks:
				card_id_to_decks[card_id] = {i}
			else:
				card_id_to_decks[card_id].add(i)

			if card_id not in card_id_to_index:
				card_id_to_index[card_id] = card_index
				card_index_to_id[card_index] = card_id
				card_index += 1

		if len(cards) == 0:
			print(file_name, "empty file found.")
			os.remove(file_name)
		for card in cards:
			card = card.lstrip("0")
			if card == "none":
				continue
			if card not in inverted_index:
				inverted_index[card] = SparseVector()
			inverted_index[card][i] = 1

		number_of_decks += 1

	for card_id in inverted_index.copy():
		if len(card_id_to_decks[card_id]) < 10:
			for deck_id in card_id_to_decks[card_id].copy():
				deck_to_card_ids[deck_id].remove(card_id)
			del card_id_to_decks[card_id]
			del inverted_index[card_id]

	for card_id in inverted_index:
		idf_scores[card_id] = math.log(number_of_decks / len(inverted_index[card_id]))

	for card_id in inverted_index:
		vector = inverted_index[card_id]
		inverted_index[card_id] = vector / vector.norm
"""
