
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import sys
import tqdm
import os


parser = argparse.ArgumentParser(
    description='Serialize KG'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()


def load_data(file_path, ent2idx, rel2idx):
    df = pd.read_csv(file_path, sep='\t', header=None)

    M = df.shape[0]  # dataset size

    X = np.zeros([M, 3], dtype=int)

    for i, row in tqdm.tqdm(df.iterrows()):
        X[i, 0] = ent2idx[row[0].lower()]
        X[i, 1] = rel2idx[row[1].lower()]
        X[i, 2] = ent2idx[row[2].lower()]

    return X


dataset_dir = 'data/{}'.format(args.dataset.rstrip('/'))
bin_dir = '{}/bin'.format(dataset_dir)

if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)

# Load dictionary
vocab_e = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_e1')
vocab_r = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_rel')

ent2idx = vocab_e[0]
rel2idx = vocab_r[0]

train_path = '{}/train.txt'.format(dataset_dir)
val_path = '{}/valid.txt'.format(dataset_dir)
test_path = '{}/test.txt'.format(dataset_dir)

X_train = load_data(train_path, ent2idx, rel2idx).astype(np.int32)
X_val = load_data(val_path, ent2idx, rel2idx).astype(np.int32)
X_test = load_data(test_path, ent2idx, rel2idx).astype(np.int32)

# Save preprocessed data
np.save('{}/train.npy'.format(bin_dir), X_train)
np.save('{}/val.npy'.format(bin_dir), X_val)
np.save('{}/test.npy'.format(bin_dir), X_test)
