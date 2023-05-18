import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
import dataset
def main(biology_data):


    """ Load data """
    sentences = treebank.sents()
    tagged_sentences = treebank.tagged_sents()

    train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2, random_state=42)



    # UNCOMMENT if you want biology dataset
    
    #train_tags = biology_data["train_tags"]
    test_tags = biology_data["test_tags"]
    #train_sentences = biology_data["train_sentences"]
    test_sentences = biology_data["test_sentences"]
  

    """ Accuracy of Uni-, Bi- and Trigram parser """
    #trigram_tagger = nltk.tag.TnT() 
    #trigram_tagger.train(train_tags) 
    
    unigram_tagger= nltk.UnigramTagger(train_tags)
    mod_accuracy, unmod_accuracy =n_gram_eval(test_sentences, unigram_tagger, test_tags)
    print(mod_accuracy, unmod_accuracy)

    """ Accuracy of max entropy model """
    #max_entropy(test_sentences, test_tags, True)

def n_gram_eval(test_sentences, tagger, test_tags):
    counter =0
    num_total =0
    num_correct=0
    for i, sentence in enumerate(test_sentences):
        py_tag = tagger.tag(sentence)
        num_total += len(py_tag)
        for y in range(len(py_tag)):
            if test_tags[i][y][1] in ['-NONE-', '#', '-LRB-', '-RRB-']:
                
                counter +=1
                num_total -=1
                continue
            print(py_tag)
            if py_tag[y][1] == test_tags[i][y][1]:
                num_correct += 1

    accuracy = tagger.accuracy(test_tags)
    return num_correct/num_total, accuracy

def max_entropy(sentences, tagged_sentences, modified):
    num_total =0
    counter=0
    num_correct=0
    for i, sentence in enumerate(sentences):
        py_tag = nltk.pos_tag(sentence)
        num_total += len(py_tag)
        for y in range(len(py_tag)):
            if tagged_sentences[i][y][1] in ['-NONE-', '#', '-LRB-', '-RRB-'] and modified:
                counter +=1
                num_total -=1
                print(py_tag[y], tagged_sentences[i][y][1])
                continue
            if py_tag[y][1] == tagged_sentences[i][y][1]:
                num_correct += 1
            

    print('Accuracy: ', num_correct/num_total)





""" Garden path """
"""
garden_path = ['The old man the boat.', 'The horse raced past the barn fell.', 'The complex houses married and single soldiers and their families.', 'The man whistling tunes pianos.', 'The cotton clothing is made of grows in Mississippi.', 'The girl told the story cried.', 'The police are trying to stop drinking and driving accidents.', 'The saw that was used to cut the board was dull.', 'The hunters shot the deer was in the forest.', 'Time flies like an arrow; fruit flies like a banana.']

for g_path in garden_path:
    print(f"{g_path}:     {nltk.pos_tag(g_path.split(' '))}")

"""



if __name__ == '__main__':
    biology_data = dataset.get_biology()

    main(biology_data)