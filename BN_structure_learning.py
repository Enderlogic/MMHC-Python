import pandas as pd
# import csv
from mmpc import mmpc_forward, mmpc_backward, asymmetry_test
from hc import hc

# data_training = pd.read_csv('Input/trainingData.csv')
data_training = pd.read_csv('Input/alarm5000.csv')
# data_training = csv.reader(open("Input/alarm5000.csv"), delimiter=";")


variables_name = data_training.keys()

state = {}

count = 0

# extract data from original file and make a list for variables
for i in data_training:
    state_temp = []

    for j in range(data_training[i].size):
        if not (data_training[i][j] in state_temp):
            state_temp.append(data_training[i][j])
    state[i] = state_temp

pc = mmpc_forward(data_training, state)
pc = mmpc_backward(pc, data_training, state)

pc = asymmetry_test(pc)

dag = hc(data_training, pc, state, 'bdeu')

a = 1