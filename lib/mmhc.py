from lib.mmpc import mmpc_forward, mmpc_backward, symmetry
from lib.hc import hc
import time
import numpy as np

def mmhc(data, test = None, score = None, prune = True, threshold = 0.05):
    '''
    mmhc algorithm
    :param data: input data (pandas dataframe)
    :param test: type of independence test (currently support g-test (for discrete data), z-test (for continuous data))
    :param score: type of score function (currently support bic (for both discrete and continuous data))
    :param prune: whether use prune method
    :param threshold: threshold for CI test
    :return: the DAG learned from data (bnlearn format)
    '''

    # initialise pc set as empty for all variables
    pc = {}

    # initialise the candidate set for variables
    can = {}
    for tar in data:
        can[tar] = list(data.columns)
        can[tar].remove(tar)

    # preprocess the data
    varnames = list(data.columns)
    if all(data[var].dtype.name == 'category' for var in data):
        arities = np.array(data.nunique())
        data = data.apply(lambda x: x.cat.codes).to_numpy()
        if test is None:
            test = 'g-test'
        if score is None:
            score = 'bic'
    elif all(data[var].dtype.name != 'category' for var in data):
        arities = None
        if test is None:
            test = 'z-test'
        if score is None or score == 'bic':
            score = 'bic_g'
    else:
        raise Exception('Mixed data is not supported.')

    # run MMPC on each variable
    start = time.time()
    for tar in varnames:
        # forward phase
        pc[tar] = []
        pc[tar], can = mmpc_forward(tar, pc[tar], can, data, arities, varnames, prune, test, threshold)
        # backward phase
        if pc[tar]:
            pc[tar], can = mmpc_backward(tar, pc[tar], can, data, arities, varnames, prune, test, threshold)
    # check the symmetry of pc set
    # when the number of variables is large, this function may be computational costly
    # this function can be merged into the pruning process during forward and backward mmpc by transmitting the whole
    # pc set into mmpc_forward and mmpc_backward
    pc = symmetry(pc)
    print('MMPC phase costs %.2f seconds' % (time.time() - start))
    # run hill-climbing
    start = time.time()
    dag = hc(data, arities, varnames, pc, score)
    print('HC phase costs %.2f seconds' % (time.time() - start))
    return dag