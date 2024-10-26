import glob
import csv

"""
['Non-Meta Decks', 'Fun/Casual Decks', 'Master Duel Decks', 'Tournament Meta Decks', 'Goat Format Decks', 
'Common Charity Decks', 'Meta Decks', 'Progression Series', 'Trinity Format Decks', 'Anime Decks', 
'Speed Duel Decks', 'Theorycrafting Decks', 'Edison Format Decks', 'Domain Format Decks', 
'Tournament Meta Decks OCG', 'Tournament Meta Decks Worlds', 'Worlds Format Decks', 'meta decks', 
'World Championship Decks', 'Legacy of the Duelist 1st Gen', 'Structure Decks']
"""

meta_formats = ["Meta Decks", "meta decks", 'Tournament Meta Decks']
non_meta_formats = ["Non-Meta Decks"]
casual_formats = ["Fun/Casual Decks", "Anime Decks"]
master_duel_formats = ["Master Duel Decks", "Tournament Meta Decks OCG", "Tournament Meta Decks Worlds",
                       "Worlds Format Decks", "World Championship Decks"]
other_formats = ["Structure Decks", "Domain Format Decks", "Edison Format Decks", "Theorycrafting Decks",
                 "Trinity Format Decks", 'Progression Series', 'Common Charity Decks', 'Goat Format Decks']

dataset_format = "other"

if dataset_format == "meta":
    allowed_formats = meta_formats
    folder_name = "meta/"

elif dataset_format == "non-meta":
    allowed_formats = non_meta_formats
    folder_name = "non_meta/"

elif dataset_format == "casual":
    allowed_formats = casual_formats
    folder_name = "casual/"

elif dataset_format == "master_duel":
    allowed_formats = master_duel_formats
    folder_name = "master_duel/"

elif dataset_format == "other":
    allowed_formats = other_formats
    folder_name = "other/"


def load_files():
    deck_number = 1
    for file_name in glob.glob("archive/*.csv"):
        with open(file_name, 'r') as file:
            csv_file = csv.reader(file)
            for i, lines in enumerate(csv_file):
                if i == 0:
                    continue
                deck_name = lines[2]
                deck_format = lines[5]

                if deck_format not in allowed_formats:
                    continue
                main_deck = string_list_to_list_of_strings(lines[6])
                extra_deck = string_list_to_list_of_strings(lines[7])
                side_deck = string_list_to_list_of_strings(lines[8])
                deck_string = "#main\n" + "\n".join(main_deck) + "\n#extra\n" + "\n".join(extra_deck) + "\n!side\n" + "\n".join(side_deck)
                with open("dataset/" + folder_name + str(deck_number) + ".ydk", "w") as deck_file:
                    deck_file.write(deck_string)
                deck_number += 1


def string_list_to_list_of_strings(array):
    return [x[1:-1].lstrip("0") for x in array[1:-1].split(",")]


load_files()
