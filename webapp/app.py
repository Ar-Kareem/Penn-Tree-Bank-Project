from flask import Flask
from flask import request
from flask import send_from_directory

from datetime import datetime
import logging

import torch
from .. import sbert

app = Flask(__name__)

logger = logging.getLogger(__name__)

def setup_logger():
    # suppress werkzeug logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    logger.setLevel(logging.DEBUG)
    # add timestamp to log
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # log to file alongside stdout
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
# RUN logger setup
setup_logger()


sbert_model = None
sbert_tokenizer = None
model = None
all_pos = None

def setup():
    global sbert_model
    global sbert_tokenizer
    global model
    global all_pos
    device = 'cpu'

    from .. import dataset
    all_pos = dataset.get_pos()

    print('Loading SBERT')
    from transformers import AutoTokenizer, AutoModel
    sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = sbert_model.to(device)

    saved_model_data = torch.load('../best_model.pt')
    model = sbert.SimpleModel(saved_model_data['config'])
    model.load_state_dict(saved_model_data['params'])
    model.eval()

# RUN SETUP
setup()

def get_pos(sentance):
    sentances = [sentance.split(' ')]
    embeds, attn_masks = sbert.sbert_encode_batched(sbert_model, sbert_tokenizer, [' '.join(x) for x in sentances], 64, verbose=False)
    embeds_pooled = sbert.pool_tokens(sentances, embeds, attn_masks, sbert_tokenizer, verbose=False)
    with torch.no_grad():
        new_embeds_out = [model(x) for x in embeds_pooled]

    for i in range(len(sentances)):
        o = new_embeds_out[i]
        prob_pos, pred_pos_id = torch.softmax(o, dim=1).max(dim=1)
        pred_pos = [all_pos[x] for x in pred_pos_id]
        return pred_pos, pred_pos_id.tolist(), prob_pos.tolist()


@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
  return send_from_directory('./simple/dist/simple/', path)


@app.route('/')
def root():
  return send_from_directory('./simple/dist/simple/', 'index.html')


@app.route('/api/', methods=['POST'])
def runpos():
    data = request.get_json()['text'].strip()
    logger.info(str(request.remote_addr) + ' | ' + str(data))
    if len(data) > 1000:
        return {
            'error': 'Text is too long',
        }
    result = get_pos(data)
    return {
        'result_pos': result[0],
        'result_pos_id': result[1],
        'result_prob': result[2],
        'result_sent': data.split(' '),
    }

# flask --app app --host=0.0.0.0 --port 47316 run