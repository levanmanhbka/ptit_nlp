import os
import re
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.data.metrics import bleu_score
import spacy
import dill as pickle
import pandas as pd
from tqdm import tqdm

from seq2seq_model import Encoder, Decoder, Seq2Seq

# === Tokenizer ===

class tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"""\n\\…\+\-\/\=\(\)‘•:\[\]\|'\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        if sentence == "":
            return []
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

# === Data Loading ===

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def read_data(src_file, trg_file):
    src_data = open(src_file).read().strip().split('\n')
    trg_data = open(trg_file).read().strip().split('\n')
    return src_data, trg_data

def create_fields(src_lang, trg_lang):
    print("loading spacy tokenizers...")
    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    return SRC, TRG

def create_dataset(src_data, trg_data, max_strlen, batchsize, device, SRC, TRG, istrain=True):
    print("creating dataset and iterator... ")
    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]
    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    data_iter = MyIterator(dataset, batch_size=batchsize, device=device,
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=istrain, shuffle=True)
    os.remove('translate_transformer_temp.csv')
    if istrain:
        SRC.build_vocab(dataset)
        TRG.build_vocab(dataset)
    return data_iter

# === Training Functions ===

def train_step(model, optimizer, criterion, batch, clip, teacher_forcing_ratio):
    model.train()
    src = batch.src.cuda()
    trg = batch.trg.cuda()
    optimizer.zero_grad()
    output = model(src, trg, teacher_forcing_ratio)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    trg = trg[1:].view(-1)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()

def evaluate(model, criterion, valid_iter):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in valid_iter:
            src = batch.src.cuda()
            trg = batch.trg.cuda()
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            total_loss.append(loss.item())
    return np.mean(total_loss)

# === Inference ===

def translate_sentence(sentence, model, SRC, TRG, device, max_len):
    model.eval()
    src_tokens = SRC.preprocess(sentence)
    src_indexes = [SRC.vocab.stoi[t] for t in src_tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [TRG.vocab.stoi['<sos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            prediction, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = prediction.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == TRG.vocab.stoi['<eos>']:
            break

    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes[1:]]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    return ' '.join(trg_tokens)

def calculate_bleu(valid_src_data, valid_trg_data, model, SRC, TRG, device, max_len):
    pred_sents = []
    for sentence in tqdm(valid_src_data, desc="Calculating BLEU score"):
        pred_trg = translate_sentence(sentence, model, SRC, TRG, device, max_len)
        pred_sents.append(pred_trg)
    pred_sents = [sent.split() for sent in pred_sents]
    trg_sents = [[sent.split()] for sent in valid_trg_data]
    return bleu_score(pred_sents, trg_sents)

# === Configuration ===

opt = {
    'train_src_data': './data/iwslt2015/train-en-vi/train.en',
    'train_trg_data': './data/iwslt2015/train-en-vi/train.vi',
    'valid_src_data': './data/iwslt2015/dev-2012-en-vi/tst2012.en',
    'valid_trg_data': './data/iwslt2015/dev-2012-en-vi/tst2012.vi',
    'src_lang': 'en_core_web_sm',
    'trg_lang': 'vi_core_news_lg',
    'max_strlen': 160,
    'batchsize': 1500,
    'device': 'cuda',
    'encoder_embedding_dim': 256,
    'decoder_embedding_dim': 256,
    'hidden_dim': 512,
    'n_layers': 2,
    'encoder_dropout': 0.5,
    'decoder_dropout': 0.5,
    'lr': 0.001,
    'epochs': 20,
    'clip': 1,
    'teacher_forcing_ratio': 0.5,
    'printevery': 200,
}

# === Load Data ===

train_src_data, train_trg_data = read_data(opt['train_src_data'], opt['train_trg_data'])
valid_src_data, valid_trg_data = read_data(opt['valid_src_data'], opt['valid_trg_data'])

SRC, TRG = create_fields(opt['src_lang'], opt['trg_lang'])
train_iter = create_dataset(train_src_data, train_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], SRC, TRG, istrain=True)
valid_iter = create_dataset(valid_src_data, valid_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], SRC, TRG, istrain=False)

# === Create Model ===

device = torch.device(opt['device'] if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    len(SRC.vocab),
    opt['encoder_embedding_dim'],
    opt['hidden_dim'],
    opt['n_layers'],
    opt['encoder_dropout'],
)

decoder = Decoder(
    len(TRG.vocab),
    opt['decoder_embedding_dim'],
    opt['hidden_dim'],
    opt['n_layers'],
    opt['decoder_dropout'],
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")

# === Optimizer & Loss ===

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
TRG_PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# === Training Loop ===

train_losses = []
valid_losses = []

for epoch in range(opt['epochs']):
    epoch_start_time = time.time()
    total_loss = 0
    iteration = 0

    for i, batch in tqdm(enumerate(train_iter), desc=f"Training Epoch {epoch}"):
        loss = train_step(model, optimizer, criterion, batch, opt['clip'], opt['teacher_forcing_ratio'])
        total_loss += loss
        iteration += 1

        if (i + 1) % opt['printevery'] == 0:
            avg_loss = total_loss / opt['printevery']
            print(f'epoch: {epoch:03d} - iter: {i:05d} - train loss: {avg_loss:.4f}')
            total_loss = 0

    train_losses.append(total_loss / max(iteration, 1))

    valid_loss = evaluate(model, criterion, valid_iter)
    valid_losses.append(valid_loss)

    bleuscore = calculate_bleu(
        valid_src_data[:500], valid_trg_data[:500],
        model, SRC, TRG, device, opt['max_strlen']
    )

    elapsed = time.time() - epoch_start_time
    print(f'epoch: {epoch:03d} - valid loss: {valid_loss:.4f} - bleu score: {bleuscore:.4f} - time: {elapsed:.4f}')

    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/seq2seq_epoch_{epoch}.pth')

# === Save Losses ===

os.makedirs('./models', exist_ok=True)
with open('./models/seq2seq_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
with open('./models/seq2seq_valid_losses.pkl', 'wb') as f:
    pickle.dump(valid_losses, f)

# === Plot Loss Curves ===

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Seq2Seq Training and Validation Loss Curves')
plt.legend()
plt.savefig('./models/seq2seq_loss_curves.png')
plt.show()

# === Final Evaluation ===

bleuscore = calculate_bleu(valid_src_data, valid_trg_data, model, SRC, TRG, device, opt['max_strlen'])
print(f'BLEU score on validation set: {bleuscore:.4f}')

# === Test Translation ===

sentence = 'I like to play football with my friends in weekend'
trans_sent = translate_sentence(sentence, model, SRC, TRG, device, opt['max_strlen'])
print(f'Translation: {trans_sent}')