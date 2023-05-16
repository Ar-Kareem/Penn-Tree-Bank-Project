
import os
import random



def get_treebank_3914():
    import nltk
    nltk.data.path.append(os.path.abspath('./nltk_data'))
    from nltk.corpus import treebank
    from sklearn.model_selection import train_test_split


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


def get_biology():
    import json
    train = json.load(open('./.data/biology/biology_data_train.json'))
    val = json.load(open('./.data/biology/biology_data_val.json'))
    test = json.load(open('./.data/biology/biology_data_test.json'))

    # filter len = 0
    train = [sent for sent in train if len(sent['data']) > 0]
    val = [sent for sent in val if len(sent['data']) > 0]
    test = [sent for sent in test if len(sent['data']) > 0]

    # shuffle 
    r = random.Random(42)
    r.shuffle(train)
    r.shuffle(val)
    r.shuffle(test)

    data = {
        'train_sentences': [[i[0] for i in sent['data']] for sent in train],
        'val_sentences': [[i[0] for i in sent['data']] for sent in val],
        'test_sentences': [[i[0] for i in sent['data']] for sent in test],
        'train_tags': [sent['data'] for sent in train],
        'val_tags': [sent['data'] for sent in val],
        'test_tags': [sent['data'] for sent in test],
        'train_meta': [{k:v for k,v in sent.items() if k != 'data'} for sent in train],
        'val_meta': [{k:v for k,v in sent.items() if k != 'data'} for sent in val],
        'test_meta': [{k:v for k,v in sent.items() if k != 'data'} for sent in test],
    }
    

    # all pos
    all_pos_train = set(x[1] for sent in data['train_tags'] for x in sent)
    all_pos_val = set(x[1] for sent in data['val_tags'] for x in sent)
    all_pos_test = set(x[1] for sent in data['test_tags'] for x in sent)
    assert len(all_pos_val.difference(all_pos_train)) == 0, 'POS in val but not in train'
    assert len(all_pos_test.difference(all_pos_train)) == 0, 'POS in test but not in train'
    
    data['all_pos'] = sorted(list(all_pos_train))
    return data

def get_pos():
    return ['#', '$', "''", ',', '-LRB-', '-NONE-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']