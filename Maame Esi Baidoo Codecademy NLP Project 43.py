from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# Setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Print out a document from each friend:
#print(goldman_docs[0:140])
#print(henson_docs[0:140])
#print(wu_docs[0:140])

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""
#Changing mystery_postcard to a writing from Emma Goldman to see how the classifier defined later holds up
#mystery_postcard = """
#It was easy enough for her to believe John Most's claim in Die Freiheit (which chance had brought her way) that Parsons, Spies, and the other defendants were to be hanged for nothing more than their advocacy of anarchism. What this doctrine was she did not quite know, but she assumed it must have merit since it favored poor workers like herself. When the jury found the men guilty, she could not accept the reality of the dread verdict. Her thoughts clung to the condemned anarchists as if they were her brothers. In her passionate yearning to do something in their behalf she attended meetings of protest and read everything she could find on the case; and she sympathetically experienced the torment of a prisoner awaiting execution.
#"""
#Changing mystery_postcard to a personal email to see how the classifier still holds up
#mystery_postcard = """
#If you don’t recognize this sign in, you can sign out of all sessions on all devices by navigating to “Security” in your Medium settings page, and clicking “Sign out other sessions”.If this was you, you can safely ignore this email.
#"""

# Create bow_vectorizer:
#Defining bow_vectorizer as an implementation of CountVectorizer
bow_vectorizer = CountVectorizer()
# Define friends_vectors:
#Using both fit and transform to train and vectorize all the writings
friends_vectors = bow_vectorizer.fit_transform(friends_docs)
# Define mystery_vector: 
#Assigning mystery_vector to the vectorized form of mystery_postcard
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Define friends_classifier as an implementation of MultinomialNB
friends_classifier = MultinomialNB()

# Train the classifier:
friends_classifier.fit(friends_vectors, friends_labels)
# Change predictions:
#Predicting which friend wrote mystery_postcard
predictions = friends_classifier.predict(mystery_vector)

mystery_friend = predictions[0] if predictions[0] else "someone else"

# Uncomment the print statement:
print("The postcard was from {}!".format(mystery_friend))
#Shows postcard is from Henson
#Finding the probabilities of each friend writing the mystery_postcard
predictions_two = friends_classifier.predict_proba(mystery_vector)
print(predictions_two)
#Probability is higher for Henson