import pandas as pd
from mmhc import mmhc
from graphviz import Digraph
from evaluation import compare
import json

# input data for learning
data_set = 'alarm'
data_size = '5000'
data_training = pd.read_csv('Input/' + data_set + data_size + '.csv')
print('data loaded successfully')

# learn bayesian network from data
dag = mmhc(data_training, score_function = 'bic', prune = True, threshold = 0.05)

# plot the graph
dot = Digraph()
for k, v in dag.items():
    if k not in dot.body:
        dot.node(k)
    if v:
        for v_ele in v:
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
compare_result = compare(true_dag, dag)
print('Compare the edges with true graph:')
print('\n'.join("{}: {}".format(k, v) for k, v in compare_result.items()))