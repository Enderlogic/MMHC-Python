import random
import urllib.request
from os import path

import pandas as pd
from graphviz import Digraph
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from lib.evaluation import f1
from lib.mmhc import mmhc

pandas2ri.activate()
base, bnlearn = importr('base'), importr('bnlearn')

# load network
network = 'alarm'
network_path = 'Input/' + network + '.rds'
if not path.isfile(network_path):
    url = 'https://www.bnlearn.com/bnrepository/' + network + '/' + network + '.rds'
    urllib.request.urlretrieve(url, network_path)
dag_true = base.readRDS(network_path)

# generate data
datasize = 10000
filename = 'Input/' + network + '_' + str(datasize) + '.csv'
if path.isfile(filename):
    data = pd.read_csv(filename, dtype='category') # change dtype = 'float64'/'category' if data is continuous/categorical
else:
    data = bnlearn.rbn(dag_true, datasize)
    data = data[random.sample(list(data.columns), data.shape[1])]
    data.to_csv(filename, index=False)


# learn bayesian network from data
dag_learned = mmhc(data)

# plot the learned graph
dot = Digraph()
for node in bnlearn.nodes(dag_learned):
    dot.node(node)
    for parent in bnlearn.parents(dag_learned, node):
        dot.edge(node, parent)
dot.render('output/' + network  + '_' + str(datasize) + '.gv', view = False)

# evaluate the learned graph
print('f1 score is ' + str(f1(dag_true, dag_learned)))
print('shd score is ' + str(bnlearn.shd(bnlearn.cpdag(dag_true), dag_learned)[0]))