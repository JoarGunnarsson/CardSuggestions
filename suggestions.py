import glob
import math
import random
import time
from SparseVec import SparseVector
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import numpy as np
import pickle
import requests
import json
import os


def load_intersection_dict():
	with open("intersection_dict.txt", "rb") as intersection_file:
		return pickle.load(intersection_file)


def download_card_data():
	response = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php?")
	content = response.content.decode("UTF-8")
	json_content = json.loads(content)
	data = json_content["data"]
	with open("dataset/info/card_data.txt", "wb") as pickle_file:
		pickle.dump(data, pickle_file)


def get_all_cards():
	response = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php?")
	content = response.content.decode("UTF-8")
	json_content = json.loads(content)
	data = json_content["data"]
	all_card_ids = []
	for card_data in data:
		card_id = str(card_data["id"])
		all_card_ids.append(card_id)
		card_images = card_data["card_images"]
		if len(card_images) > 1:
			for alt_card in card_images[1:]:
				alt_card_id = str(alt_card["id"])
				all_card_ids.append(alt_card_id)

	all_card_ids.sort()
	x = {}
	for card_id in all_card_ids:
		x[card_id] = True
	all_card_ids = x
	with open("dataset/card_list.txt", "wb") as card_file:
		pickle.dump(all_card_ids, card_file)


def load_all_cards():
	with open("dataset/card_list.txt", "rb") as card_file:
		card_array = pickle.load(card_file)
	return card_array


def clean_file(file_name, all_card_list):
	with open(file_name, "r") as deck_file:
		lines = deck_file.read().split("\n")
		lines = [x for x in lines if "#" in x or "!" in x or x in all_card_list]

	with open(file_name, "w") as deck_file:
		content = "\n".join(lines)
		deck_file.write(content)


def index_decks(directory):
	number_of_decks = 0
	card_index = 0
	for i, file_name in enumerate(glob.glob(f"{directory}*.ydk", recursive=True)):
		if (i+1) % 1000 == 0:
			print("Progress indexing files:", i+1, "files indexed.")

		cards = load_deck(file_name)
		random.shuffle(cards)
		corpus.append(cards)

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

	"""
	for card_id in inverted_index.copy():
		if len(card_id_to_decks[card_id]) < 10:
			for deck_id in card_id_to_decks[card_id].copy():
				deck_to_card_ids[deck_id].remove(card_id)
			del card_id_to_decks[card_id]
			del inverted_index[card_id]
	"""
	for card_id in inverted_index:
		idf_scores[card_id] = math.log(number_of_decks / len(inverted_index[card_id]))

	for card_id in inverted_index:
		vector = inverted_index[card_id]
		inverted_index[card_id] = vector / vector.norm


def use_context_to_update_index():
	iters = 1
	for j in range(iters):
		new_inverted_index = {}
		for i, card_id in enumerate(inverted_index):
			if i % 100 == 0:
				print(i+1)
			this_vec = np.zeros((534,))
			for index in card_id_to_context_vector[card_id]:
				this_vec = this_vec + card_id_to_context_vector[card_id][index] * card_id_to_archetype_vector[card_index_to_id[index]]
			sorted_vec = this_vec.copy()
			sorted_vec.sort()
			clipping_value = sorted_vec[-5]
			this_vec[this_vec < clipping_value] = 0
			new_inverted_index[card_id] = this_vec
		for card_id in new_inverted_index:
			card_id_to_archetype_vector[card_id] = new_inverted_index[card_id] / np.linalg.norm(new_inverted_index[card_id])


def load_context_vectors():
	global card_id_to_context_vector
	with open("context_vectors.txt", "rb") as pickle_file:
		card_id_to_context_vector = pickle.load(pickle_file)


def create_context_vectors():
	for i, card_id in enumerate(inverted_index):
		if (i+1) % 1000 == 0:
			print(i+1)
		card_vector = SparseVector()
		decks_containing_card = card_id_to_decks[card_id]
		for deck_id in decks_containing_card:
			neighboring_cards = deck_to_card_ids[deck_id]
			for neighbor in neighboring_cards:
				neighbor_index = card_id_to_index[neighbor]
				card_vector.increment(neighbor_index, 1)
		card_vector[card_id_to_index[card_id]] = 1
		card_id_to_context_vector[card_id] = card_vector / card_vector.norm
	#with open("CardSuggestions/context_vectors.txt", "wb") as pickle_file:
	#pickle.dump(card_id_to_context_vector, pickle_file)


def create_intersection_dict():
	print(len(inverted_index))
	for i, card_id in enumerate(inverted_index.keys()):
		if i % 1000 == 0:
			print(i)
		for card_id2 in inverted_index:
			compute_intersection(card_id, card_id2)

	f = open("intersection_dict.txt", "wb")
	pickle.dump(intersection_dict, f)
	f.close()


def load_deck(file_name):
	with open(file_name, "r") as f:
		deck_file_lines = f.read().split("\n")
	cards = clean_cards(deck_file_lines)
	return list(set(cards))


def filter_deck(cards):
	cards = clean_cards(cards)
	cards = [x for x in cards if x in inverted_index]
	return cards


def clean_cards(cards):
	return [x.lstrip("0") for x in cards if x != "" and "#" not in x and "!" not in x and "none" not in x]


def cosine_similarity(p1, p2, idf_score, dot_function):
	return dot_function(p1, p2) * idf_score


def cosine_distance(p1, p2):
	return 1 - SparseVector.dot(p1, p2) / p1.norm


def suggest_cards_knn(deck_list, model, n=10, re_suggest=False):
	deck_list = filter_deck(deck_list)
	deck_vector = model.mean_vector(deck_list)

	return knn_cosine_similarity(deck_vector, model, n, re_suggest, deck_list)


def get_idf_score(card_id):
	return idf_scores[card_id]


def knn_cosine_similarity(deck_vector, model, n, re_suggest, deck_list=None):
	# TODO: allowed cards - exactly what is it, look in index or somewhere else?
	similarity = {}
	for _card_id in allowed_cards:
		idf_score = model.get_idf_score(_card_id)
		similarity[_card_id] = cosine_similarity(deck_vector, model.get_vector(_card_id), idf_score, model.get_dot_function())

	suggestions = allowed_cards.copy()

	suggestions.sort(key=lambda x: score_similarity(x, deck_list, similarity, re_suggest), reverse=True)
	top_suggestions = suggestions[:n]
	return top_suggestions


def compute_average_deck_vector(deck_list):
	deck_vector = SparseVector()
	for card_id in deck_list:
		if card_id not in inverted_index:
			continue
		deck_vector = deck_vector + inverted_index[card_id]
	deck_vector = deck_vector / len(deck_list)
	return deck_vector


def score_similarity(x, deck_list, similarity, re_suggest):
	if not re_suggest and x in deck_list:
		return -10 ** 10
	return similarity[x]


def suggest_cards(deck_list, n=10, re_suggest=False):
	return suggest_cards_knn(deck_list, n, re_suggest)


def suggest_cards_from_file(file_name, n=10):
	deck_list = load_deck(file_name)
	return suggest_cards(deck_list, n=n)


def request_deck(number=9800):
	from urllib.request import Request, urlopen

	req = Request(
		url=f'https://yugiohtopdecks.com/ygopro_deck/{number}',
		headers={'User-Agent': 'Mozilla/5.0'}
	)
	webpage = urlopen(req).read()
	with open(f"./decks/{number}.ydk", "wb") as file:
		file.write(webpage)


def download_all_decks():
	for i in range(1, 10000):
		print(i, i / 10000)
		request_deck(i)
		time.sleep(0.1)


def assign_cards_to_closest_cluster(deck_list, cluster_centers):
	cluster_members = [[] for _ in cluster_centers]
	for card_id in deck_list:
		card_postings = inverted_index[card_id]
		saved_cluster_index = find_closest_cluster(card_postings, cluster_centers)
		cluster_members[saved_cluster_index].append((card_id, card_postings))
	return cluster_members


def find_closest_cluster(card_postings, cluster_centers):
	center_postings_0 = cluster_centers[0]
	min_distance = cosine_distance(center_postings_0, card_postings)
	saved_cluster_index = 0
	for i, center in enumerate(cluster_centers[1:]):
		center_postings = cluster_centers[i + 1]
		distance = cosine_distance(center_postings, card_postings)
		if distance < min_distance:
			min_distance = distance
			saved_cluster_index = i + 1
	return saved_cluster_index


def find_second_closest_cluster(card_postings, cluster_centers):
	center_postings_0 = cluster_centers[0]
	min_distance = cosine_distance(center_postings_0, card_postings)
	min_cluster_index = 0
	for i, center in enumerate(cluster_centers[1:]):
		center_postings = cluster_centers[i + 1]
		distance = cosine_distance(center_postings, card_postings)
		if distance < min_distance:
			min_distance = distance
			min_cluster_index = i + 1

	min_distance = cosine_distance(center_postings_0, card_postings)
	saved_cluster_index = 1
	for i, center in enumerate(cluster_centers[1:]):
		center_postings = cluster_centers[i + 1]
		distance = cosine_distance(center_postings, card_postings)
		if distance < min_distance and i + 1 != min_cluster_index:
			min_distance = distance
			saved_cluster_index = i + 1
	return saved_cluster_index


def k_means_clustering(deck_list, k=5):
	if len(deck_list) < k:
		k = len(deck_list)
	iters = 10
	cluster_center_cards = random.sample(deck_list, k)
	cluster_centers = [inverted_index[center_card_id] for center_card_id in
					   cluster_center_cards]

	cluster_members = []

	for j in range(iters):
		cluster_centers, cluster_members = one_k_means_iteration(deck_list, cluster_centers)

	squared_distance_sum = 1 - compute_silhouette_score(deck_list, cluster_centers,
														cluster_members)  # compute_total_cluster_variation(cluster_centers, cluster_members)
	## Is this better?
	return cluster_centers, cluster_members, squared_distance_sum


def compute_silhouette_score(deck_list, cluster_centers, cluster_members):
	s = 0
	for card_id in deck_list:
		card_postings = inverted_index[card_id]
		cluster_index = find_closest_cluster(card_postings, cluster_centers)
		a = 0
		for cluster_member in cluster_members[cluster_index]:
			if cluster_member[0] == card_id:
				continue
			a += cosine_distance(card_postings, cluster_member[1]) ** 2

		second_cluster_index = find_second_closest_cluster(card_postings, cluster_centers)

		b = 0
		for cluster_member in cluster_members[second_cluster_index]:
			if cluster_member[0] == card_id:
				continue
			b += cosine_distance(card_postings, cluster_member[1]) ** 2

		if a != 0 or b != 0:
			s += (b - a) / max(a, b)

	return s / len(deck_list)


def one_k_means_iteration(deck_list, cluster_centers):
	cluster_members = assign_cards_to_closest_cluster(deck_list, cluster_centers)
	for i, center in enumerate(cluster_centers):
		center_postings = center
		for cluster_member in cluster_members[i]:
			center_postings = center_postings + cluster_member[1]

		if len(cluster_members[i]) != 0:
			center_postings = center_postings / len(cluster_members[i])
		cluster_centers[i] = center_postings

	return cluster_centers, cluster_members


def compute_total_cluster_variation(cluster_centers, cluster_members):
	squared_distance_sum = 0
	for i, center in enumerate(cluster_centers):
		center_postings = center
		for cluster_member in cluster_members[i]:
			squared_distance_sum += cosine_distance(center_postings, cluster_member[1]) ** 2
	return squared_distance_sum


def get_cluster_card_ids(cluster_centers, cluster_members):
	cluster_card_ids = []
	for i, center in enumerate(cluster_centers):
		this_cluster_cards = []
		for cluster_member in cluster_members[i]:
			this_cluster_cards.append(cluster_member[0])
		if len(this_cluster_cards) != 0:
			cluster_card_ids.append(this_cluster_cards)
	return cluster_card_ids


def find_best_k_means_cluster(deck_list, n=40, k=5):
	if len(deck_list) == 0:
		return [], 0
	saved_cluster_centers, saved_cluster_members, min_distance_sum = k_means_clustering(deck_list, k=k)
	for i in range(n - 1):
		cluster_centers, cluster_members, distance_sum = k_means_clustering(deck_list, k=k)
		if distance_sum < min_distance_sum:
			saved_cluster_centers, saved_cluster_members = cluster_centers, cluster_members
			min_distance_sum = distance_sum

	s = compute_silhouette_score(deck_list, saved_cluster_centers, saved_cluster_members)

	cluster_card_ids = get_cluster_card_ids(saved_cluster_centers, saved_cluster_members)
	return cluster_card_ids, s


def k_means(deck_list, n=10):
	start_time = time.time()
	start_k = 2
	stop_k = 10
	best_clustering, max_s = find_best_k_means_cluster(deck_list, n=n, k=start_k)
	print(2, max_s, "i, s")
	best_k = start_k
	for i in range(start_k + 1, stop_k + 1):
		clustering, s = find_best_k_means_cluster(deck_list, n=n, k=i)
		print(i, s, "i, s")
		if s > max_s:
			max_s = s
			best_clustering = clustering
			best_k = i
	print(f"The best clustering was with k = {best_k}")
	print(f"It took {time.time() - start_time} seconds.")
	return best_clustering


def print_clustering(clustering):
	for cluster in clustering:
		cluster_string = "\n".join(cluster)
		print(cluster_string)
		print("20129614")


class MyCorpus:
	"""An iterator that yields sentences (lists of str)."""

	def __iter__(self):
		for line in corpus:
			yield line


def sklearn_clustering(deck_list):
	X = []
	for card_id in deck_list:
		X.append(card2vec_model[card_id])

	saved_cluster_labels = None
	max_score = -1

	for k in range(2, 10):
		clustering_model = KMeans(n_clusters=k, n_init=30)
		cluster_labels = clustering_model.fit_predict(preprocessing.normalize(X))
		silhouette_avg = silhouette_score(X, cluster_labels)
		if silhouette_avg >= max_score:
			saved_cluster_labels = cluster_labels
	cluster_labels = saved_cluster_labels

	unique_cluster_labels = sorted(list(set(cluster_labels)))
	clustering = []
	for label in unique_cluster_labels:
		this_cluster = []
		for card_id, assigned_label in zip(deck_list, cluster_labels):
			if assigned_label == label:
				this_cluster.append(card_id)
		clustering.append(this_cluster)

	print_clustering(clustering)
	return clustering


def suggest_on_clusters(deck_cards, n=50, re_suggest=False):
	deck_cards = filter_deck(deck_cards)
	clustering = sklearn_clustering(deck_cards)
	recs = []
	for cluster_cards in clustering:
		recs.extend(suggest_cards_knn(cluster_cards, n=int(n/len(deck_cards)*len(cluster_cards)), re_suggest=re_suggest))
		recs.append("20129614")
	return recs


def suggest_context(deck_cards, n=50, re_suggest=False):
	deck_cards = filter_deck(deck_cards)

	deck_vector = SparseVector()
	for card_id in deck_cards:
		deck_vector = deck_vector + card_id_to_context_vector[card_id]
	deck_vector = deck_vector / len(deck_cards)

	scores = {}

	for card_id in inverted_index:
		card_vector = card_id_to_context_vector[card_id]
		scores[card_id] = SparseVector.dot(deck_vector, card_vector)

	sorted_scores = allowed_cards.copy()
	sorted_scores.sort(key=lambda x: scorer(x, scores, deck_cards, re_suggest), reverse=True)
	results = sorted_scores[:n]

	return results


def suggest_dot(deck_cards, n=50, re_suggest=False):
	deck_cards = [x for x in deck_cards if x in inverted_index]

	deck_vector = deck_to_id_vector(deck_cards)
	scores = {}

	for card_id in allowed_cards:
		card_vector = cards_to_encoding[card_id]
		scores[card_id] = np.dot(deck_vector, card_vector)

	sorted_scores = allowed_cards.copy()
	sorted_scores.sort(key=lambda x: scorer(x, scores, deck_cards, re_suggest), reverse=True)
	results = sorted_scores[:n]

	return results


def suggest_archetype(deck_cards, n=50, re_suggest=False):
	deck_cards = [x for x in deck_cards if x in inverted_index]
	for card_id in deck_cards:
		print(card_id, card_id_to_archetype[card_id])
	deck_vector = np.zeros((534,))
	for card_id in deck_cards:
		deck_vector += card_id_to_archetype_vector[card_id]
	deck_vector /= len(deck_cards)
	scores = {}

	for card_id in allowed_cards:
		card_vector = card_id_to_archetype_vector[card_id]
		scores[card_id] = np.dot(deck_vector, card_vector)

	sorted_scores = allowed_cards.copy()
	sorted_scores.sort(key=lambda x: scorer(x, scores, deck_cards, re_suggest), reverse=True)
	results = sorted_scores[:n]

	for card_id in results:
		print(card_id, card_id_to_archetype[card_id], scorer(card_id, scores, deck_cards, False))

	vec = deck_vector# card_id_to_archetype_vector["34800281"]
	arch = [id_to_archetype[i] for i, idk in enumerate(vec) if vec[i] != 0]
	arch_score = [idk for i, idk in enumerate(vec) if vec[i] != 0]
	for arche, score in zip(arch, arch_score):
		print(arche, score)
	return results


def suggest_prob(deck_cards, n=50, re_suggest=False):
	deck_cards = filter_deck(deck_cards)
	score = {}
	card_scores = compute_prob(deck_cards)
	for i, card_id in enumerate(allowed_cards):
		score[card_id] = card_scores[i] * get_idf_score(card_id)

	results = allowed_cards.copy()

	results.sort(key=lambda x: scorer(x, score, deck_cards, re_suggest), reverse=True)
	results = results[:n]
	return results


def compute_prob(deck_cards):
	probabilities = []
	for card_id in allowed_cards:
		number_of_decks_containing_given_card = len(card_id_to_decks[card_id])
		log_prior = np.log(number_of_decks_containing_given_card+1) - np.log(len(deck_to_card_ids.keys())+1)

		card_probs = []
		for deck_card in deck_cards:
			counter = get_intersection(deck_card, card_id)
			this_prob = np.log(counter+1)
			card_probs.append(this_prob)

		card_probs = np.array(card_probs) - np.log(number_of_decks_containing_given_card + 1)
		probabilities.append(np.mean(card_probs) + log_prior)

	probabilities = np.array(probabilities)
	return probabilities - np.min(probabilities)


def compute_intersection(deck_card, given_card):
	hash_id = min(deck_card, given_card) + "," + max(deck_card, given_card)
	if hash_id in intersection_dict:
		return
	decks_given = card_id_to_decks[given_card]
	decks_this_deck_card = card_id_to_decks[deck_card]
	counter = len(decks_given.intersection(decks_this_deck_card))
	if counter == 0:
		return
	intersection_dict[hash_id] = counter


def get_intersection(deck_card, given_card):
	hash_id = min(deck_card, given_card) + "," + max(deck_card, given_card)
	if hash_id not in intersection_dict:
		return 0
	return intersection_dict[hash_id]


def scorer(card_id, scores, deck_list, re_suggest):
	if not re_suggest and card_id in deck_list:
		return -10 ** 10
	return scores[card_id]


def id_embedding():
	model = load_model().wv
	for card_id in inverted_index:
		cards_to_encoding[card_id] = np.array(model[card_id])
	return model


def load_model():
	return Word2Vec.load("models/card2vec.model")


def train_model():
	vector_dimension = 100
	window = 90
	model = Word2Vec(sentences=MyCorpus(), vector_size=vector_dimension, window=window, min_count=0)
	model.save("/models/card2vec.model")
	return model


def deck_to_id_vector(deck_cards):
	return np.array(card2vec_model.get_mean_vector(deck_cards))


DECK_DIRECTORY = "dataset/tournaments/"
inverted_index = {}
corpus = []
cards_to_encoding = {}
card_id_to_decks = {}
deck_to_card_ids = {}
card_id_to_name = {}
intersection_dict = load_intersection_dict()
idf_scores = {}
card_id_to_context_vector = {}
card_id_to_index = {}
card_index_to_id = {}

index_decks(DECK_DIRECTORY)

with open("dataset/info/card_data.txt", "rb") as file:
	all_card_data = pickle.load(file)


archetype_to_index = {}
index_to_archetype = {}
card_id_to_archetype = {}
id_to_archetype = {}
card_id_to_archetype_vector = {}
for card_data in all_card_data:
	card_id = str(card_data["id"])
	if "archetype" in card_data:
		archetype = card_data["archetype"]
	else:
		archetype = None

	card_id_to_archetype[card_id] = archetype

archetype_index = 0
num_archetypes = 534
for card_id in inverted_index:

	if card_id not in card_id_to_archetype:
		archetype = None
		card_id_to_archetype[card_id] = None
	else:
		archetype = card_id_to_archetype[card_id]

	if archetype not in archetype_to_index and archetype is not None:
		archetype_to_index[archetype] = archetype_index
		id_to_archetype[archetype_index] = archetype
		archetype_index += 1


for card_id in inverted_index:
	archetype = card_id_to_archetype[card_id]
	archetype_vector = np.zeros((num_archetypes,))
	if archetype is not None:
		archetype_index = archetype_to_index[archetype]
		archetype_vector[archetype_index] = 1
	card_id_to_archetype_vector[card_id] = archetype_vector


card2vec_model = id_embedding()
load_context_vectors()

with open("lflist.conf", "r") as file:
	content = file.read()
	lines = content.split("\n")
	allowed_cards = [x.split(" ")[0] for x in lines if "!" not in x and "$" not in x]
	allowed_cards = [x for x in allowed_cards if x in inverted_index]

allowed_cards = list(inverted_index.keys())
