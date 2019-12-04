import numpy as np
from scipy.stats.distributions import chi2
import operator
import itertools

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
            contingency_table = count_number(tar, pc, can_var, data)
            # compute p-value for each candidate variable
            p_value[can_var] = max(p_value_calculator(contingency_table, pc), p_value[can_var])
        else:
            for r in range(len(pc)):
                for pc_sub in itertools.combinations(pc[0 : -1], r):

                    # don't check the separating set checked before
                    pc_con = list(pc_sub)
                    pc_con.append(pc[-1])
                    # count the value in contingency table
                    contingency_table = count_number(tar, pc_con, can_var, data)

                    # compute p-value for each candidate variable
                    p_value[can_var] = max(p_value_calculator(contingency_table, pc_con), p_value[can_var])
    return p_value

# contingency table calculator
def count_number(tar, pc, can, data):

    # initialise the contingency table
    contingency_table = {}
    contingency_table['S_kij'] = {}
    contingency_table['S_kj'] = {}

    state = {}
    for i in data:
        state[i] = np.unique(data[i]).tolist()

    if pc:
        # initialise each individual space in contingency table
        for key_pc, value_pc in data.groupby(pc):
            contingency_table['S_kij'][key_pc] = {}
            contingency_table['S_kj'][key_pc] = {}

            for key_tar, value_tar in value_pc.groupby(tar):
                contingency_table['S_kij'][key_pc][key_tar] = {k: v.shape[0] for k, v in value_tar.groupby(can)}

            contingency_table['S_kj'][key_pc] = {k: v.shape[0] for k, v in value_pc.groupby(can)}

            # make sure every situation is recorded including 0 appearance
            for tar_state in state[tar]:
                if tar_state not in contingency_table['S_kij'][key_pc]:
                    contingency_table['S_kij'][key_pc][tar_state] = {}
                for can_state in state[can]:
                    if can_state not in contingency_table['S_kij'][key_pc][tar_state]:
                        contingency_table['S_kij'][key_pc][tar_state][can_state] = 0

            for can_state in state[can]:
                if can_state not in contingency_table['S_kj'][key_pc]:
                    contingency_table['S_kj'][key_pc][can_state] = 0
    else:
        # initialise each individual space in contingency table
        for key_tar, value_tar in data.groupby(tar):
            contingency_table['S_kij'][key_tar] = {k: v.shape[0] for k,v in value_tar.groupby(can)}

        # make sure every situation is recorded including 0 appearance
        for tar_state in state[tar]:
            if tar_state not in contingency_table['S_kij']:
                contingency_table['S_kij'][tar_state] = {}
            for can_state in state[can]:
                if can_state not in contingency_table['S_kij'][tar_state]:
                    contingency_table['S_kij'][tar_state][can_state] = 0

        contingency_table['S_kj'] = {k: v.shape[0] for k, v in data.groupby(can)}
    return contingency_table

# p-value calculator (based on G-test)
def p_value_calculator(contingency_table, pc):

    G = 0
    if pc:
        for key_pc in contingency_table['S_kij']:
            for key_tar in contingency_table['S_kij'][key_pc]:
                for key_can in contingency_table['S_kij'][key_pc][key_tar]:
                    if contingency_table['S_kij'][key_pc][key_tar][key_can] != 0:
                        G = G + 2 * contingency_table['S_kij'][key_pc][key_tar][key_can] * np.log(
                            contingency_table['S_kij'][key_pc][key_tar][key_can] * sum(contingency_table['S_kj'][key_pc].values()) /
                            sum(contingency_table['S_kij'][key_pc][key_tar].values()) / contingency_table['S_kj'][key_pc][key_can])

        dof = len(contingency_table['S_kij']) * (len(contingency_table['S_kij'][key_pc]) - 1) * (
                    len(contingency_table['S_kj'][key_pc]) - 1)
    else:
        for key_tar in contingency_table['S_kij']:
            for key_can in contingency_table['S_kij'][key_tar]:
                if contingency_table['S_kij'][key_tar][key_can] != 0:
                    G = G + 2 * contingency_table['S_kij'][key_tar][key_can] * np.log(
                        contingency_table['S_kij'][key_tar][key_can] * sum(contingency_table['S_kj'].values()) /
                        sum(contingency_table['S_kij'][key_tar].values()) / contingency_table['S_kj'][key_can])

        dof = (len(contingency_table['S_kij']) - 1) * (len(contingency_table['S_kj']) - 1)

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