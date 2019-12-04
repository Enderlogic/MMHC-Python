from mmpc import mmpc_forward, mmpc_backward, symmetry
from hc import hc
import time

def mmhc(data, score_function = 'bdeu', prune = False, threshold = 0.05):
    # data: input training data
    # score: the type of score function, currently support 'bdeu'
    # prune: prune candidate variable by previous results
    # threshold: threshold for CI test

    # initialise pc set as empty for all variables
    pc = {}

    # initialise the candidate set for variables
    can = {}
    for tar in data:
        can[tar] = list(data.columns)
        can[tar].remove(tar)

    start_time = time.time()
    # run MMPC on each variable
    for tar in data:
        # forward phase
        pc[tar] = []
        pc[tar], can = mmpc_forward(tar, pc[tar], can, data, prune, threshold)
        # backward phase
        pc[tar], can = mmpc_backward(tar, pc[tar], can, data, prune, threshold)

    end_time = time.time()
    print("run time for mmpc:", end_time - start_time, "seconds")
    # check the symmetry of pc set
    # when the number of variables is large, this function may be computational costly
    # this function can be merged into the pruning process during forward and backward mmpc by transmitting the whole
    # pc set into mmpc_forward and mmpc_backward
    pc = symmetry(pc)

    start_time = time.time()
    # run hill-climbing
    dag = hc(data, pc, score_function)
    end_time = time.time()
    print("run time for hc:", end_time - start_time, "seconds")
    return dag