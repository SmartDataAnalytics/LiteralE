import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import spacy
from pathlib import Path
import unicodedata


parser = argparse.ArgumentParser(
    description='Create text literals'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()


# Load vocab
vocab = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_e1', allow_pickle=True)

ent2idx = vocab[0]
idx2ent = vocab[1]

# Load raw literals
df = pd.read_csv(f'data/{args.dataset}/literals/text_literals.txt', header=None, sep='\t')

# Load preprocessor
nlp = spacy.load('en_core_web_md')

txt_lit = np.zeros([len(ent2idx), 300], dtype=np.float32)
cnt = 0


for ent, txt in tqdm(zip(df[0].values, df[2].values)):
    key = unicodedata.normalize('NFC', ent.lower())
    idx = ent2idx.get(key)

    if idx is not None:
        txt_lit[idx, :] = nlp(txt).vector
    else:
        cnt += 1


print(f'Ignoring {cnt} texts.')
print('Saving text features of size {}'.format(txt_lit.shape))
np.save(f'data/{args.dataset}/literals/text_literals.npy', txt_lit)
