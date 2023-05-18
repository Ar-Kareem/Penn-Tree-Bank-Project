import json
import argparse

from tqdm import tqdm
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = type('wandb', (object,), {'log': lambda *args, **kwargs: None, 'init': lambda *args, **kwargs: None})


def sbert_encode(model, tokenizer, inp_sentences):
    """
    inp_sentences: list[str], list of sentences to encode
    tokenizer: AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    model: AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    """

    tokens = {'input_ids': [], 'attention_mask': []}

    with torch.no_grad():
        for sentence in inp_sentences:
            new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        tokens['input_ids'] = torch.stack(tokens['input_ids']).to(model.device)
        tokens['attention_mask'] = torch.stack(tokens['attention_mask']).to(model.device)

        # get embeddings
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.detach()  # torch.Size([batch, 128, 768])
    return embeddings, tokens['attention_mask']

def sbert_encode_batched(model, tokenizer, inp_sentences, batch_size, verbose=True):
    embeds, attn_masks = [], []
    pbar = tqdm if verbose else lambda x: x
    for i in pbar(range(0, len(inp_sentences), batch_size)):
        x = inp_sentences[i:i+batch_size]
        embed, attn_mask = sbert_encode(model, tokenizer, x)
        embeds.append(embed.cpu())
        attn_masks.append(attn_mask.cpu())
    embeds = torch.cat(embeds, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)
    return embeds, attn_masks


def __join_s2(lst):
    return ''.join([x if not x.startswith('##') else x[2:] for x in lst])

def __get_s1_to_s2(s1, s2):
    s1 = [x.lower() for x in s1]
    s2 = [x.lower() for x in s2]
    s1_to_s2 = {i:[] for i in range(len(s1))}
    i = 0
    j = 0
    while i < len(s1) or j < len(s2):
        s1_i_nospace = s1[i].replace(' ', '')  # biology data has spaces in a single token
        for delta in range(1, len(s2)+1):
            if s1_i_nospace == __join_s2(s2[j:j+delta]):
                s1_to_s2[i].extend(range(j, j+delta))
                # [s2_to_s1.append(i) for _ in range(delta)]
                i += 1
                j += delta
                break
        else:
            print('ERROR\n', s1, '\n', s2)
            return {}
    assert i == len(s1) and j == len(s2)
    return s1_to_s2

def pool_tokens(data, embeds, attn_masks, tokenizer, verbose=True):
    result = []
    pbar = tqdm if verbose else lambda x: x
    for data_i in pbar(range(len(data))):
        s1 = data[data_i]
        s2 = tokenizer.convert_ids_to_tokens(tokenizer.encode_plus(' '.join(data[data_i]))['input_ids'])[1:-1]
        s1_to_s2 = __get_s1_to_s2(s1, s2)

        e = embeds[data_i, attn_masks[data_i]==1]
        begin_embed = e[0]
        e = e[1:]
        res = []
        for i, j in s1_to_s2.items():
            # print(s1[i], '|'.join(s2[k] for k in j))
            if max(j) >= e.shape[0]:  # in case tokenizer exceeds max length
                res.append(begin_embed)  # add the dummy token instead
                continue
            res.append(e[j].mean(dim=0))
        res = torch.stack(res)
        result.append(res)
    return result

def single_epoch(model, dataset, criterion_ce, epoch, optim=None, is_train=True, device='cpu', pos_mapper=None, skip_pos=None, verbose=True):
    if is_train:
        model.train()
    else:
        model.eval()
    metrics = {'acc_correct': 0, 'total': 0, 'losses': []}
    pbar = tqdm(dataset) if verbose else dataset
    for batch_idx, batch in enumerate(pbar):
        embeds, sents, pos_text, pos_tag = zip(*batch)
        embeds = torch.cat(embeds, dim=0).to(device)
        pos_tag = torch.tensor([n for sent in pos_tag for n in sent], device=device)
        assert embeds.shape[0] == pos_tag.shape[0], f'{embeds.shape[0]} != {pos_tag.shape[0]}'

        with torch.set_grad_enabled(is_train):
            out = model(embeds)
        
        if pos_mapper is not None:
            out = pos_mapper(out)

        loss = criterion_ce(out, pos_tag)
        metrics['losses'].append(loss.item())
        metrics['acc_correct'] += (out.argmax(dim=1) == pos_tag).sum().item()
        metrics['total'] += pos_tag.shape[0]

        if is_train and optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        if verbose:
            pbar.set_description(f'{"Train" if is_train else "Val"} Epoch: {epoch} '\
                             f'Acc: {100*metrics["acc_correct"]/metrics["total"]:.1f}% '\
                             f'Loss: {np.mean(metrics["losses"]) :.6f}')

    loss = np.mean(metrics["losses"])
    acc = 100*metrics["acc_correct"]/metrics["total"]
    if wandb.run is not None:
        if is_train:
            wandb.log({"train_loss": loss, "train_acc": acc})
        else:
            wandb.log({"val_loss": loss, "val_acc": acc})
    return acc

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, embeds, tags, all_pos, skip_pos=None):
        self.embeds = embeds
        if skip_pos is None:
            self.sents = [[n[0] for n in sent] for sent in tags]
            self.pos_text = [[n[1] for n in sent] for sent in tags]
            self.pos_tag = [[all_pos.index(pos) for w,pos in sent] for sent in tags]
            self.data = list(zip(self.embeds, self.sents, self.pos_text, self.pos_tag))
        else:
            self.sents = [[] for _ in tags]
            self.pos_text = [[] for _ in tags]
            self.pos_tag = [[] for _ in tags]
            embeds_after_skip = [[] for _ in tags]
            for sent_i, sent in enumerate(tags):
                for word_i, (word, pos) in enumerate(sent):
                    if pos not in skip_pos:
                        self.sents[sent_i].append(word)
                        self.pos_text[sent_i].append(pos)
                        self.pos_tag[sent_i].append(all_pos.index(pos))
                        embeds_after_skip[sent_i].append(embeds[sent_i][word_i])
            self.embeds = [torch.stack(l) for l in embeds_after_skip]
            self.data = list(zip(self.embeds, self.sents, self.pos_text, self.pos_tag))
        # check that all sentences are consistent
        for i, d in enumerate(self.data):
            assert len(d[0]) == len(d[1]) == len(d[2]) == len(d[3]), (i, (len(d[0]), len(d[1]), len(d[2]), len(d[3])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SimpleModel(torch.nn.Module):
    def __init__(self, model_config):
        super(SimpleModel, self).__init__()
        self.model_config = model_config
        if 'dropout' in model_config and model_config['dropout'] > 0:
            self.dropout = torch.nn.Dropout(p=model_config['dropout'])
        if 'hidden_dims' in model_config:
            self.hidden_dims = [model_config['input_dim']] + model_config['hidden_dims'] + [model_config['output_dim']]
            self.layers = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]) for i in range(1, len(self.hidden_dims))])
        else:
            self.layers = torch.nn.Linear(model_config['input_dim'], model_config['output_dim'])

    def forward(self, x):
        if isinstance(self.layers, torch.nn.Linear):
            return self.layers(x)
        else:
            for layer in self.layers[:-1]:
                x = torch.nn.functional.relu(layer(x))
                if hasattr(self, 'dropout'):
                    x = self.dropout(x)
            return self.layers[-1](x)

def read_json_prop(prop, filename='sbert.json'):
    with open(filename, 'r') as f:
        return json.load(f)[prop]
def write_json_prop(prop, value, filename='sbert.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    data[prop] = value
    with open(filename, 'w') as f:
        json.dump(data, f)


def train_pipeline(args):
    device = args.device
    epochs = args.epochs

    import dataset
    print('Getting data')
    data = dataset.get_treebank_3914()

    print('Loading SBERT')
    from transformers import AutoTokenizer, AutoModel
    sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = sbert_model.to(device)

    print('Getting embeddings from SBERT')
    embeds, attn_masks = sbert_encode_batched(sbert_model, sbert_tokenizer, [' '.join(x) for x in data['train_sentences']], 64)
    train_embeds_pooled = pool_tokens(data['train_sentences'], embeds, attn_masks, sbert_tokenizer)
    embeds, attn_masks = sbert_encode_batched(sbert_model, sbert_tokenizer, [' '.join(x) for x in data['test_sentences']], 64)
    test_embeds_pooled = pool_tokens(data['test_sentences'], embeds, attn_masks, sbert_tokenizer)
    del sbert_model

    train_dataset = ListDataset(train_embeds_pooled, data['train_tags'], data['all_pos'])
    test_dataset = ListDataset(test_embeds_pooled, data['test_tags'], data['all_pos'])
    batch_size = 64
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    print('Preparing model')
    model_config = {
        'input_dim': 768, 
        # 'hidden_dims': [256],
        'output_dim': len(data['all_pos']),
        # 'dropout': 0.5,
    }
    model = SimpleModel(model_config).to(device)
    param_count = sum([p.numel() for n,p in model.named_parameters()])
    print(f'trainable #p: {param_count:,}')

    lr = 1e-3
    weight_decay = 0.0001
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_ce = torch.nn.CrossEntropyLoss()
    cur_epoch = 0

    wandb.init(project="6.8630 Penn Treebank Project", config={
        "architecture": "SimpleModel",
        "dataset": "treebank_3914",
        "learning_rate": lr,
        'weight_decay': weight_decay,
        "model_config": model_config,
        "#params": param_count,
    })
    for _ in range(int(epochs)):
        train_acc = single_epoch(model, train_dataset, criterion_ce, epoch=cur_epoch, optim=optim, is_train=True, device=device)
        val_acc = single_epoch(model, test_dataset, criterion_ce, epoch=cur_epoch, optim=None, is_train=False, device=device)
        if val_acc > float(read_json_prop('best_val')):
            write_json_prop('best_val', val_acc)
            torch.save(model.state_dict(), 'best_model.pt')
            print('---saved best model---')
        cur_epoch += 1

def inference_pipeline(args):
    device = args.device
    new_samples = args.new_samples

    input_from_cmd = new_samples is None
    sentances = new_samples.split(',') if new_samples else []

    import dataset
    all_pos = dataset.get_pos()

    print('Loading SBERT')
    from transformers import AutoTokenizer, AutoModel
    sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sbert_model = sbert_model.to(device)

    saved_model_data = torch.load('best_model.pt')
    model = SimpleModel(saved_model_data['config'])
    model.load_state_dict(saved_model_data['params'])

    while True:
        if input_from_cmd:
            sentances = [input('Enter a sentances: ')]

        sentances = [x.split(' ') for x in sentances]
        embeds, attn_masks = sbert_encode_batched(sbert_model, sbert_tokenizer, [' '.join(x) for x in sentances], 64, verbose=False)
        embeds_pooled = pool_tokens(sentances, embeds, attn_masks, sbert_tokenizer, verbose=False)
        with torch.no_grad():
            new_embeds_out = [model(x) for x in embeds_pooled]

        for i in range(len(sentances)):
            o = new_embeds_out[i]
            prob_pos, pred_pos_id = torch.softmax(o, dim=1).max(dim=1)
            pred_pos = [all_pos[x] for x in pred_pos_id]
            print('sent', sentances[i])
            print('pred', pred_pos)
            print('prob', [str(round(100*x.item()))+'%' for x in prob_pos])
            print('')
        if not input_from_cmd:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'sbert')
    parser.set_defaults(func=lambda _: parser.print_help())  # in case no argument is passed print help
    root_subparsers = parser.add_subparsers(help="Sub Commands Help")

    # SUBPARSER train
    subparser = root_subparsers.add_parser('train', help="Run train pipeline")
    subparser.add_argument("--device", help="Choose device. Default: cpu", default='cpu')
    subparser.add_argument("--epochs", help="# of epochs to train, default: 100", default=100)
    subparser.set_defaults(func=train_pipeline)

    # SUBPARSER inference
    subparser = root_subparsers.add_parser('inference', help="Run inference pipeline")
    subparser.add_argument("--device", help="Choose device. Default: cpu", default='cpu')
    subparser.add_argument("--new_samples", help="the sentances to try, default: input from command line", default=None)
    subparser.set_defaults(func=inference_pipeline)

    # FOR DEBUGGING
    sys.argv = ['sbert.py', 'inference']

    args = parser.parse_args()
    args.func(args)
