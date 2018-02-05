# LiteralE
Knowledge Graph Embeddings learned from the structure and literals of knowledge graphs.

This work is built on top of Tim Dettmers' ConvE codes: <https://github.com/TimDettmers/ConvE>.
Thus credit goes to Tim Dettmers for the base implementation.

### Getting Started

1. Install PyTorch
2. Preprocess datasets: `chmod +x preprocess.sh && ./preprocess.sh && python wrangle_KG.py`
3. Preprocess literals: `python preprocess_num_lit.py`


### Reproducing Paper's Experiments

For DistMult+LiteralE and ComplEx+LiteralE:
```
python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} model {DistMult, ComplEx} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True
```

For ConvE+LiteralE:
```
python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True
```

NB: For base models, replace `main_literal.py` with `main.py`.