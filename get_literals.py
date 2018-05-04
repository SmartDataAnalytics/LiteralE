import re, time, pickle
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound, EndPointInternalError
import pandas as pd
import pdb

############################################################
# Get data to represent tokens from target formal language #
############################################################

class Querier(object):
    def __init__(self, address="http://drogon:8890/sparql", **kw):
        # address = "http://localhost:9890/sparql"        # TODO remote testing
        super(Querier, self).__init__(**kw)
        self.sparql = SPARQLWrapper(address)
        self.sparql.setReturnFormat(JSON)

    def _exec_query(self, query):
        self.sparql.setQuery(query)
        retries = 10
        while True:
            try:
                res = self.sparql.query().convert()
                return res
            except (EndPointInternalError, EndPointNotFound) as e:
                print("retrying {}".format(retries))
                retries -= 1
                if retries < 0:
                    raise e
                time.sleep(1)

    def get_triples_of(self, entity, language=None):

        #query = """SELECT DISTINCT ?p ?o WHERE {{
        #    {} ?p ?o .
        #    {}
        #}}""".format(entity, "FILTER (!isLiteral(?o) || lang(?o) = \"\" || langMatches(lang(?o), '{}'))".format(language) if language is not None else "")
        
        query = """SELECT DISTINCT ?p ?o WHERE {{
            {} ?p ?o .
            {}
        }}""".format(entity, "FILTER (isLiteral(?o))")
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        alltriples = []
        literaltriples = []
        uritriples = []
        for result in results:
            p = result["p"]["value"]
            o = result["o"]["value"]
            o_type = result["o"]["type"]
            if o_type == "literal":
                literaltriples.append((entity, p, o))
            elif o_type == "typed-literal":
                literaltriples.append((entity, p, o, result["o"]["datatype"]))
            elif o_type == "uri":
                uritriples.append((entity, p, o))
            else:
                raise q.SumTingWongException("o type: {}".format(o_type))
            alltriples.append((entity, p, o))
        return alltriples, uritriples, literaltriples


if __name__ == "__main__":

    filename = '../data/fb15k/freebase_mtr100_mte100-train.txt'
    data = pd.read_csv(filename, header=None, sep='\t') 
    entities_ = list(set(data[0].values).union(set(data[2].values)))
    print('Total Entities:', len(entities_))

    que = Querier()
    # Literals for Subject
    f = open('../data/fb15k-literal/fb15k_numerical_triples.txt','w')    
    for i, entity in enumerate(entities_):
        entity = entity[1:].replace('/','.')
        print('Querying for entity {}'.format(entity))
        alltriples, uritriples, literaltriples = que.get_triples_of("<http://rdf.freebase.com/ns/"+ entity+">")
        print('Extracting Numerical Attributes for entity {}'.format(entity))
        numerical_literaltriples = []
        for triple in literaltriples:
            try:
                literal = float(triple[2])
                numerical_literaltriples.append(triple)
            except ValueError:
                continue
        for triple in numerical_literaltriples:
            triple = '\t'.join(en for en in triple)
            f.write(triple +'\n')
    f.close()