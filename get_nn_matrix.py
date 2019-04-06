import torch
import torch.nn.functional as F
import numpy as np
from model import Config, DistMultLiteral_gate
from scipy.spatial.distance import squareform, pdist

# DistMult
weights = torch.load('pretrained_models/FB15k-237_DistMult_0.2_0.0.model', lambda storage, loc: storage)
E = weights['emb_e.weight'].numpy()
dists = squareform(pdist(E))
vals = np.sort(dists)[:, :10]
idxs = np.argsort(dists)[:, :10]
np.save('results/NN/distmult_nn_vals.npy', vals)
np.save('results/NN/distmult_nn_idxs.npy', idxs)


# KBLN
weights = torch.load('pretrained_models/FB15k-237_KBLN_0.2_0.0_literal.model', lambda storage, loc: storage)
E = weights['emb_e.weight'].numpy()
dists = squareform(pdist(E))
vals = np.sort(dists)[:, :10]
idxs = np.argsort(dists)[:, :10]
np.save('results/NN/kbln_nn_vals.npy', vals)
np.save('results/NN/kbln_nn_idxs.npy', idxs)


# MTKGNN
weights = torch.load('pretrained_models/FB15k-237_MTKGNN_0.2_0.0_literal.model', lambda storage, loc: storage)
E = weights['emb_e.weight'].numpy()
dists = squareform(pdist(E))
vals = np.sort(dists)[:, :10]
idxs = np.argsort(dists)[:, :10]
np.save('results/NN/mtkgnn_nn_vals.npy', vals)
np.save('results/NN/mtkgnn_nn_idxs.npy', idxs)


# Literals only
num_lits = np.load(f'data/FB15k-237/literals/numerical_literals.npy')
max_lit, min_lit = np.max(num_lits, axis=0), np.min(num_lits, axis=0)
num_lits = (num_lits - min_lit) / (max_lit - min_lit + 1e-8)
dists = squareform(pdist(num_lits))
vals = np.sort(dists)[:, :10]
idxs = np.argsort(dists)[:, :10]
np.save('results/NN/literals_only_nn_vals.npy', vals)
np.save('results/NN/literals_only_nn_idxs.npy', idxs)


# LiteralE
params = torch.load('pretrained_models/FB15k-237_DistMultLiteral_gate_0.2_0.0_literal.model', lambda storage, loc: storage)

Config.embedding_dim = 200
model = DistMultLiteral_gate(num_lits.shape[0], 239, num_lits)
model.load_state_dict(params)
model.cuda()

E = model.emb_num_lit(model.emb_e.weight, model.numerical_literals).cpu().data.numpy()
dists = squareform(pdist(E))
vals = np.sort(dists)[:, :10]
idxs = np.argsort(dists)[:, :10]
np.save('results/NN/literale_nn_vals.npy', vals)
np.save('results/NN/literale_nn_idxs.npy', idxs)
