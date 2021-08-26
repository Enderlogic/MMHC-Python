import pandas as pd
import random
from graphviz import Digraph
from os import path
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from lib.evaluation import f1
from lib.mmhc import mmhc

pandas2ri.activate()
base, bnlearn = importr('base'), importr('bnlearn')

# load network
network = 'alarm'
dag_true = base.readRDS('Input/' + network + '.rds')

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
dag_learned = mmhc(data, prune = True, threshold = 0.05)

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