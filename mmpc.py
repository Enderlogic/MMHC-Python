import numpy as np
from scipy.stats.distributions import chi2
import operator
import itertools
import time

# forward phase of MMPC
def mmpc_forward(tar, pc, can, data, prune, threshold):
    # tar: target variable
    # pc: parents and children set of the target variable
    # can: candidate variable for the pc set of the target variable
    # data: input data
    # prune: prune the target variable in the candidate variable set for those variables which are independent with the target one

    # run until no candidate variable for current variable
    p_value = {}
    for can_var in can[tar]:
        p_value[can_var] = 0

    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test(p_value, tar, pc, can[tar], data)
        # update pc set and candidate set
        pc, can = update_forward(p_value, tar, pc, can, prune, threshold)
    return pc, can

# backward phase of MMPC
def mmpc_backward(tar, pc, can, data, prune, threshold):
    # tar: target variable
    # pc: parents and children set of the target variable
    # can: candidate variable for the pc set of the target variable
    # data: input data
    # prune: prune the target variable in the candidate variable set for those variables which are independent with the target one

    # transfer the variable in pc set to candidate set except the last one
    can[tar] = pc[0: -1]
    pc_output = []
    pc_output.append(pc[-1])
    can[tar].reverse()

    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test({}, tar, pc_output, can[tar], data)
        # update pc set and candidate set
        pc_output, can = update_backward(p_value, tar, pc_output, can, prune, threshold)

    return pc_output, can

# conditional independence test
def independence_test(p_value, tar, pc, can, data):
    for can_var in can:
        if can_var not in p_value.keys():
            p_value[can_var] = 0
        if len(pc) == 0:
            # count the value in contingency table
            # start_time = time.time()
            N_kij, dof = count_number(tar, pc, can_var, data[[tar, can_var]])
            # end_time1 = time.time()
            # print(end_time1 - start_time)
            # compute p-value for each candidate variable
            p_value[can_var] = max(p_value_calculator(N_kij, pc, dof), p_value[can_var])
            # end_time2 = time.time()
            # print('pc = 0, count_number:', end_time1 - start_time, 's')
            # print('pc = 0, p_value_calculator:', end_time2 - end_time1, 's')
        else:
            for r in range(len(pc)):
                for pc_sub in itertools.combinations(pc[0 : -1], r):
                    # don't check the separating set checked before
                    pc_con = list(pc_sub)
                    pc_con.append(pc[-1])
                    # count the value in contingency table
                    # start_time = time.time()
                    contingency_table, dof = count_number(tar, pc_con, can_var, data[[tar, can_var] + pc_con])
                    # end_time1 = time.time()
                    # compute p-value for each candidate variable
                    p_value[can_var] = max(p_value_calculator(contingency_table, pc_con, dof), p_value[can_var])
                    # end_time2 = time.time()
                    # print('pc != 0, count_number:', end_time1 - start_time, 's')
                    # print('pc != 0 , p_value_calculator:', end_time2 - end_time1, 's')
    return p_value

# contingency table calculator
def count_number(tar, pc, can, data):
    if pc:
        N_kij = {}
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        for pc_var in pc:
            dof = dof * data[pc_var].nunique()
        for key_pc, value_pc in data.groupby(pc):
            N_kij[key_pc] = {k: v[can].value_counts().to_dict() for k, v in value_pc.groupby(tar)}
    else:
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        N_kij = {k: v[can].value_counts().to_dict() for k, v in data.groupby(tar)}
    return N_kij, dof

# p-value calculator (based on G-test)
def p_value_calculator(N_kij, pc, dof):
    G = 0
    if pc:
        for key_pc in N_kij:
            N_kj = {}
            for key_tar in N_kij[key_pc]:
                N_kj = {k : N_kj.get(k, 0) + N_kij[key_pc][key_tar].get(k, 0) for k in set(N_kj) | set(N_kij[key_pc][key_tar])}
            for key_tar in N_kij[key_pc]:
                for key_can in N_kij[key_pc][key_tar]:
                    G = G + 2 * N_kij[key_pc][key_tar][key_can] * np.log(N_kij[key_pc][key_tar][key_can] * sum(N_kj.values()) /
                        sum(N_kij[key_pc][key_tar].values()) / N_kj[key_can])
    else:
        N_kj = {}
        for key_tar in N_kij:
            N_kj = {k : N_kj.get(k, 0) + N_kij[key_tar].get(k, 0) for k in set(N_kj) | set(N_kij[key_tar])}
        for key_tar in N_kij:
            for key_can in N_kij[key_tar]:
                G = G + 2 * N_kij[key_tar][key_can] * np.log(N_kij[key_tar][key_can] * sum(N_kj.values()) /sum(N_kij[key_tar].values()) / N_kj[key_can])

    p_value = chi2.sf(G, dof)
    return p_value

# pc and candidate set update function for forward phase
def update_forward(p_value, tar, pc, can, prune, threshold):

    # add the variable with lowest p-value to pc set and remove it from the candidate set
    sorted_p_value = sorted(p_value.items(), key = operator.itemgetter(1))

    if sorted_p_value[0][1] <= threshold:
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

# pc and candidate set update function for backward phase
def update_backward(p_value, tar, pc, can, prune, threshold):

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