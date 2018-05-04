import pandas as pd
from collections import Counter

# Read Literal DataSet
filename = 'data/fb15k-literal/fb15k_numerical_triples.txt'
triples = [line.strip().split('\t') for line in open(filename,'r')]
all_predicate = list(zip(*triples))[1]
predicate_freq = Counter(all_predicate)

# Literal filtering
minm_count = 5
filtered_predicate = {predicate : count for predicate, count in predicate_freq.items() if (('http://rdf.freebase.com/key/' not in predicate) and count>minm_count)}	

filtered_triples = []
for triple in triples:
	if triple[1] in filtered_predicate:
		filtered_triples.append(triple)

with open('data/fb15k-literal/filtered-numerical-fb15k.txt','w') as f:
	for triple in filtered_triples:
		f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')