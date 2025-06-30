

def clean_cards(cards):
	"""
	Cleans a card list, removing items that are not cards and returns a new list.
	"""
	return [x.lstrip("0") for x in cards if x != "" and "#" not in x and "!" not in x and "none" not in x]


def load_deck(file_name):
	"""
	Loads a deck from a file, cleans it, and returns it.
	TODO: Note: No longer makes it a set, might be desired, look into it.
	"""
	with open(file_name, "r") as f:
		deck_file_lines = f.read().split("\n")
	cards = clean_cards(deck_file_lines)
	return cards
