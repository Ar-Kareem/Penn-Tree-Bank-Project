
import os

import nltk
nltk.data.path.append(os.path.abspath('./nltk_data'))

from nltk.corpus import treebank
from nltk.parse import RecursiveDescentParser
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger

from sklearn.model_selection import train_test_split


def get_treebank_3914():
    sentences = treebank.sents()
    parse_trees = treebank.parsed_sents()
    tagged_sentences = treebank.tagged_sents()
    train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2, random_state=42)

    # all pos
    all_pos = set(x[1] for sent in train_tags for x in sent)
    all_pos_test = set(x[1] for sent in test_tags for x in sent)
    assert len(all_pos_test.difference(all_pos)) == 0, 'POS in test but not in train'
    all_pos = sorted(list(all_pos))

    return {
        'train_sentences': train_sentences,
        'test_sentences': test_sentences,
        'train_tags': train_tags,
        'test_tags': test_tags,
        'all_pos': all_pos,
    }

def get_pos():
    return ['#', '$', "''", ',', '-LRB-', '-NONE-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']