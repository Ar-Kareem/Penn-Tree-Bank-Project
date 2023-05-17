import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split


""" Load data """
sentences = treebank.sents()
tagged_sentences = treebank.tagged_sents()


#Tosses away words with * in it
new_sentences = []
new_tagged_sentences = []




"""
for sentence, tagged_sentence in zip(sentences, tagged_sentences):
    new_sentence = []
    new_tagged_sentence = []
    for word, tagged_word in zip(sentence, tagged_sentence):
        counter +=1
        if tagged_word[1] not in ['-NONE-', '#', '-LRB-', '-RRB-']:
            counter_2 +=1
            new_sentence.append(word)
            new_tagged_sentence.append(tagged_word)
    new_sentences.append(new_sentence)
    new_tagged_sentences.append(new_tagged_sentence)


print('num_removed', counter-counter_2)

print("diff_len: ", len(new_sentences)- len(new_tagged_sentences))
print(len(new_sentences))
print(len(sentences))
"""


train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2, random_state=42)



# Prediciton and evaluation
num_correct = 0
num_total = 0

""" Accuracy of Uni-, Bi- and Trigram parser """
trigram_tagger = nltk.UnigramTagger(train_tags)

"""
counter =0
for i, sentence in enumerate(test_sentences):
    py_tag = trigram_tagger.tag(sentence)
    num_total += len(py_tag)
    for y in range(len(py_tag)):
        if test_tags[i][y][1] in ['-NONE-', '#', '-LRB-', '-RRB-']:
            counter +=1
            num_total -=1
            continue
        print(py_tag[y],test_tags[i][y])
        if py_tag[y][1] == test_tags[i][y][1]:
            num_correct += 1

            

print('Accuracy trigram: ', num_correct/num_total)
"""


accuracy = trigram_tagger.accuracy(test_tags)

print("Trigram accuracy: ", accuracy)




""" Preprocesses data """

#Tosses away words with * in it
new_sentences = []
new_tagged_sentences = []

for sentence, tagged_sentence in zip(sentences, tagged_sentences):
    new_sentence = []
    new_tagged_sentence = []
    for word, tagged_word in zip(sentence, tagged_sentence):
        if '*' not in word:
            new_sentence.append(word)
            new_tagged_sentence.append(tagged_word)
    new_sentences.append(new_sentence)
    new_tagged_sentences.append(new_tagged_sentence)


""" Evaluates the pos_dag on the accuracy data"""
# Prediciton and evaluation
num_correct = 0
num_total = 0

for i, sentence in enumerate(new_sentences):
    py_tag = nltk.pos_tag(sentence)
    num_total += len(py_tag)
    for y in range(len(py_tag)):
        if py_tag[y][1] == new_tagged_sentences[i][y][1]:
            num_correct += 1
            

print('Accuracy: ', num_correct/num_total)



""" Garden path """

garden_path = ['The old man the boat.', 'The horse raced past the barn fell.', 'The complex houses married and single soldiers and their families.', 'The man whistling tunes pianos.', 'The cotton clothing is made of grows in Mississippi.', 'The girl told the story cried.', 'The police are trying to stop drinking and driving accidents.', 'The saw that was used to cut the board was dull.', 'The hunters shot the deer was in the forest.', 'Time flies like an arrow; fruit flies like a banana.']

#for g_path in garden_path:
#    print(f"{g_path}:     {nltk.pos_tag(g_path.split(' '))}")



""" Uses classifier-based POS tagger """

from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from nltk.tag.sequential import ClassifierBasedPOSTagger

class CustomClassifierBasedPOSTagger(ClassifierBasedPOSTagger):

    def feature_detector(self, tokens, index, history):
        return {
            key: str(value) # Ensure that the feature value is a string. Converts None to 'None'
            for key, value in super().feature_detector(tokens, index, history).items()
        }

bnb = SklearnClassifier(BernoulliNB())
bnb_tagger = CustomClassifierBasedPOSTagger(train=train_tags,
                                            classifier_builder=bnb.train,
                                            verbose=True)

sentence = "This is a sample sentence which I just made for fun."
# evaluate tagger on test data and sample sentence
print(bnb_tagger.accuracy(test_tags))

# see results on our previously defined sentence
print(bnb_tagger.tag(nltk.word_tokenize(sentence)))