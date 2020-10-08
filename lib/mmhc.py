from lib.mmpc import mmpc_forward, mmpc_backward, symmetry
from lib.hc import hc
import time

def mmhc(data, score_function = 'bdeu', prune = False, threshold = 0.05):
    # input:
    # data: input training data
    # score: the type of score function, currently support 'bdeu', 'bic'
    # prune: prune candidate variable by previous results
    # threshold: threshold for CI test
    # output:
    # dag: a dictionary containing variables with their parents

    # initialise pc set as empty for all variables
    pc = {}

    # initialise the candidate set for variables
    can = {}
    for tar in data:
        can[tar] = list(data.columns)
        can[tar].remove(tar)

    start_time = time.time()
    # run MMPC on each variable
    # forward_time = 0
    # backward_time = 0
    for tar in data:
        # forward phase
        pc[tar] = []
        # start_time = time.time()
        pc[tar], can = mmpc_forward(tar, pc[tar], can, data, prune, threshold)
        # end_time = time.time()
        # forward_time = forward_time + end_time - start_time
        # print('run time for forward:', end_time - start_time, 'seconds')
        # backward phase
        # start_time = time.time()
        if pc[tar]:
            pc[tar], can = mmpc_backward(tar, pc[tar], can, data, prune, threshold)
        # end_time = time.time()
        # backward_time = backward_time + end_time - start_time
        # print('run time for backward', end_time - start_time, 'seconds')
        print(tar)

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