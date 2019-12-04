import pandas as pd
from mmhc import mmhc
from graphviz import Digraph

# input data for learning
input_file = 'alarm1000'
data_training = pd.read_csv('Input/' + input_file + '.csv')

# input data for evaluation
# (to be finishied)

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

dot.render('output/' + input_file + '.gv', view=True)
# evaluate the result
# (to be finished)