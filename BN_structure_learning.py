import pandas as pd
import numpy as np
from lib.mmhc import mmhc
from graphviz import Digraph
from lib.evaluation import compare
from lib.accessory import cpdag
import json

# input data for learning
data_set = 'alarm'
data_size = '0.1'
data_training = pd.read_csv('Input/' + data_set + data_size + '.csv')
print('data loaded successfully')

# add noise into original data
# data_noise = data_training.copy()
# epsilon = 0.1
# for var in data_noise:
#     values = data_noise[var].unique()
#     len_values = len(values)
#     noise = epsilon / (len_values - 1) * np.ones((len_values, len_values)) + (1 - epsilon * len_values / (len_values - 1)) * np.identity(len_values)
#     for val in values:
#         data_noise[var][data_noise[var] == val] = np.random.choice(values, len(data_noise[var][data_noise[var] == val]), p = noise[np.where(values == val)[0][0], :])

# learn bayesian network from data
dag = mmhc(data_training, score_function = 'bic', prune = False, threshold = 0.05)

# plot the graph
dot = Digraph()
for k, v in dag.items():
    if k not in dot.body:
        dot.node(k)
    if v:
        for v_ele in v['par']:
            if v_ele not in dot.body:
                dot.node(v_ele)
            dot.edge(v_ele, k)
dot.render('output/' + data_set + data_size + '.gv', view = False)

# save the result
with open('output/' + data_set + data_size + 'result.json', 'w') as fp:
    json.dump(dag, fp)

# evaluate the result (comment the following lines if evaluation is not necessary)
with open('Input/' + data_set + '.json') as json_file:
     true_dag = json.load(json_file)

compare_result = compare(cpdag(true_dag), cpdag(dag))
print('Compare the edges with true graph:')
print('\n'.join("{}: {}".format(k, v) for k, v in compare_result.items()))