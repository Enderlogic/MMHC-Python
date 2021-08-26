from lib.accessory import independence_test
import operator

# forward phase of MMPC
def mmpc_forward(tar, pc, can, data, arities, varnames, prune, test, threshold):
    '''
    forward phase of mmpc
    :param tar: target variable
    :param pc: parents and children set of the target variable
    :param can: candidate variable for the pc set of the target variable
    :param data: input data (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param prune: whether use prune method
    :param test: type of statistical test (currently support g-test)
    :param threshold: threshold for statistical test to determine independence
    :return: pc set and candidate pc set
    '''

    # run until no candidate variable for current variable
    p_value = {}
    for can_var in can[tar]:
        p_value[can_var] = 0

    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test(p_value, tar, pc, can[tar], data, arities, varnames, test, threshold)
        # print(p_value)
        # update pc set and candidate set
        pc, can = update_forward(p_value, tar, pc, can, prune, threshold)
    return pc, can


# backward phase of MMPC
def mmpc_backward(tar, pc, can, data, arities, varnames, prune, test, threshold):
    '''
    backward phase of mmpc
    :param tar: target variable
    :param pc: parents and children set of the target variable
    :param can: candidate variable for the pc set of the target variable
    :param data: input data (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param prune: whether use prune method
    :param test: type of statistical test (currently support g-test)
    :param threshold: threshold for statistical test to determine independence
    :return: pc set and candidate pc set
    '''

    # transfer the variable in pc set to candidate set except the last one
    can[tar] = pc[0: -1]
    pc_output = []
    pc_output.append(pc[-1])
    can[tar].reverse()

    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test({}, tar, pc_output, can[tar], data, arities, varnames, test, threshold)
        # update pc set and candidate set
        pc_output, can = update_backward(p_value, tar, pc_output, can, prune, threshold)
    return pc_output, can


def update_forward(p_value, tar, pc, can, prune, threshold):
    '''
    add the variable with lowest p-value to pc set and remove it from the candidate set
    :param p_value: a dictionary contains the maximum p-value of CI tests for each variable
    :param tar: target variable
    :param pc: parents and children set of the target variable
    :param can: candidate variable for the pc set of the target variable
    :param prune: whether use prune method
    :param threshold: threshold for statistical test to determine independence
    :return: updated pc set and candidate variables
    '''
    sorted_p_value = sorted(p_value.items(), key=operator.itemgetter(1))

    if sorted_p_value[0][1] < threshold:
        pc.append(sorted_p_value[0][0])
        can[tar].remove(sorted_p_value[0][0])
        p_value.pop(sorted_p_value[0][0], None)

    # remove independent variables from candidate set
    independent_can = [x for x in sorted_p_value if x[1] > threshold]
    for ind in independent_can:
        can[tar].remove(ind[0])
        p_value.pop(ind[0])
        # prune the target variable from the candidate set of the candidate variable if they are independent
        if prune:
            if tar in can[ind[0]]:
                can[ind[0]].remove(tar)
    return pc, can


def update_backward(p_value, tar, pc, can, prune, threshold):
    '''
    pc and candidate set update function for backward phase
    :param p_value: a dictionary contains the maximum p-value of CI tests for each variable
    :param tar: target variable
    :param pc: parents and children set of the target variable
    :param can: candidate variable for the pc set of the target variable
    :param prune: whether use prune method
    :param threshold: threshold for statistical test to determine independence
    :return: updated pc set and candidate variables
    '''
    # initialise the output candidate set
    can_output = []

    # signal of import variable
    sig_import = 1
    for can_var in can[tar]:
        if p_value[can_var] <= threshold:
            if sig_import:
                pc.append(can_var)
                sig_import = 0
            else:
                can_output.append(can_var)
        else:
            if prune:
                if tar in can[can_var]:
                    can[can_var].remove(tar)

    can[tar] = can_output

    return pc, can


# symmetry check for pc set
def symmetry(pc):
    for var in pc:
        pc_remove = []
        for par in pc[var]:
            if var not in pc[par]:
                pc_remove.append(par)
        if pc_remove:
            for par in pc_remove:
                pc[var].remove(par)
    return pc
