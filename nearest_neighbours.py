import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description='Nearest neighbours of an entity of FB15k'
)
parser.add_argument('--ent', metavar='',)
args = parser.parse_args()

distmult_nn_idx = np.load('results/NN/distmult_nn_idxs.npy')
distmult_nn_val = np.load('results/NN/distmult_nn_vals.npy')
kbln_nn_idx = np.load('results/NN/kbln_nn_idxs.npy')
kbln_nn_val = np.load('results/NN/kbln_nn_vals.npy')
mtkgnn_nn_idx = np.load('results/NN/mtkgnn_nn_idxs.npy')
mtkgnn_nn_val = np.load('results/NN/mtkgnn_nn_vals.npy')
literale_nn_val = np.load('results/NN/literale_nn_vals.npy')
literale_nn_idx = np.load('results/NN/literale_nn_idxs.npy')

vocab = np.load('vocab_e1')
ent2idx = vocab[0]
idx2ent = vocab[1]

fb2yago = np.load('fb2yago.npy').item()
yago2fb = np.load('yago2fb.npy').item()

fb_ent = yago2fb[args.ent]

print(f'Entity: {args.ent} {fb_ent}\n')

print('DistMult')
print('--------')
dm_nn_idx = distmult_nn_idx[ent2idx[fb_ent]]
dm_nn_val = distmult_nn_val[ent2idx[fb_ent]]

for idx, val in zip(dm_nn_idx, dm_nn_val):
    try:
        print(f'{fb2yago[idx2ent[idx]]}: {val:.3f}')
    except KeyError:
        print(f'{idx2ent[idx]}: {val:.3f}')


print('\nKBLN')
print('--------')
dm_nn_idx = kbln_nn_idx[ent2idx[fb_ent]]
dm_nn_val = kbln_nn_val[ent2idx[fb_ent]]

for idx, val in zip(dm_nn_idx, dm_nn_val):
    try:
        print(f'{fb2yago[idx2ent[idx]]}: {val:.3f}')
    except KeyError:
        print(f'{idx2ent[idx]}: {val:.3f}')


print('\nMTKGNN')
print('--------')
dm_nn_idx = mtkgnn_nn_idx[ent2idx[fb_ent]]
dm_nn_val = mtkgnn_nn_val[ent2idx[fb_ent]]

for idx, val in zip(dm_nn_idx, dm_nn_val):
    try:
        print(f'{fb2yago[idx2ent[idx]]}: {val:.3f}')
    except KeyError:
        print(f'{idx2ent[idx]}: {val:.3f}')


print('\nLiterals Nearest Neighbor')
print('--------------------------')
literal_nn_val = np.load('results/NN/literals_only_nn_vals.npy')
literal_nn_idx = np.load('results/NN/literals_only_nn_idxs.npy')

le_nn_idx = literal_nn_idx[ent2idx[fb_ent]]
le_nn_val = literal_nn_val[ent2idx[fb_ent]]

for idx, val in zip(le_nn_idx, le_nn_val):
    try:
        print(f'{fb2yago[idx2ent[idx]]}: {val:.3f}')
    except KeyError:
        print(f'{idx2ent[idx]}: {val:.3f}')


print('\nLiteralE')
print('--------')
le_nn_idx = literale_nn_idx[ent2idx[fb_ent]]
le_nn_val = literale_nn_val[ent2idx[fb_ent]]

for idx, val in zip(le_nn_idx, le_nn_val):
    try:
        print(f'{fb2yago[idx2ent[idx]]}: {val:.3f}')
    except KeyError:
        print(f'{idx2ent[idx]}: {val:.3f}')
