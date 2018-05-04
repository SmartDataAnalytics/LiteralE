
filename1 = 'data/FB15k/literals/numerical_literals.txt'
filename2 = 'data/FB15k/literals/FB15K_NumericalTriples.txt'
triples1 = [line.strip().split('\t') for line in open(filename1,'r')]
literal_relation = list(set(list(zip(*triples1))[1]))
literal_relation = ['<'+relation+'>' for relation in literal_relation]
triples2 = [line.strip().split('\t') for line in open(filename2,'r')]
for triple in triples2:
	if triple[1] in literal_relation:
		continue
	triples1.append(triple)	
with open('combined_literal_triples.txt','w') as f:
	for triple in triples1:
		triple = '\t'.join(en for en in triple)
		f.write(triple+'\n')

