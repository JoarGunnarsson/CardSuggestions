import suggestions
import models
import index

deck_list = ['89943723', '25451383', '25451383', '72656408', '72656408', '3717252', '48048590', '48048590', '48048590', '40044918', '68468459', '68468459', '68468459', '45484331', '45484331', '45484331', '95515789', '95515789', '19096726', '16605586', '15717011', '59438930', '8949584', '8949584', '62022479', '62022479', '44362883', '44362883', '44362883', '213326', '81439173', '75500286', '44394295', '29948294', '48130397', '48130397', '48130397', '9175957', '9175957', '17751597', '75286621', '20366274', '41209827', '69946549', '70534340', '70534340', '24915933', '41373230', '60461804', '87746184', '87746184', '34848821', '1906812', '50907446']

#index.generate_dataset("dataset/meta/", "meta")
my_index = index.InvertedIndex("Datasets/card_lists/tournaments/")
model = models.CFModel(my_index)

result = suggestions.suggest_cards_knn(deck_list, model, 30)
print("\n".join(result), "\n", "-" * 20)

corp = index.Corpus("Datasets/card_lists/tournaments/")
model = models.Card2VecModel(corp, my_index)

result = suggestions.suggest_cards_knn(deck_list,  model,30)
print("\n".join(result))


#clustering = suggestions.k_means(result)
#suggestions.print_clustering(clustering)


# TODO: For SparseVector, when an element is set to 0, remove it from the key set! Important both for speed and
# correctness for len(key_set) etc.
# TODO: fix issue when card does not exist in the inverted index - perhaps with get, but issue with vectors -
# perhaps just skip - need to look in the allowed cards.
# TODO: If only k < n good reccomendations are found - stop, don't recommend bad cards.
# TODO: Look into issue of card2vec generating popular cards - reflects bias in the dataset, how to fix?
