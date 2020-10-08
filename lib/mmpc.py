import numpy as np
from scipy.stats.distributions import chi2
import operator
import itertools
import pandas as pd
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
        print(p_value)
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
            # contingency_table, dof = count_number(tar, pc, can_var, data[[tar, can_var]])
            contingency_table, dof = count_number_compensation(tar, pc, can_var, data[[tar, can_var]], epsilon = 0.1)
            # end_time1 = time.time()
            # print(end_time1 - start_time)
            # compute p-value for each candidate variable
            p_value[can_var] = max(p_value_calculator(contingency_table, pc, dof), p_value[can_var])
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
                    # contingency_table, dof = count_number(tar, pc_con, can_var, data[[tar, can_var] + pc_con])
                    contingency_table, dof = count_number_compensation(tar, pc_con, can_var, data[[tar, can_var] + pc_con], epsilon = 0.1)
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
        N_kij_dict = {}
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        pc_all = []
        for pc_var in pc:
            dof = dof * data[pc_var].nunique()
            pc_all.append(data[pc_var].unique())
        pc_all = list(itertools.product(*pc_all))

        N_kij = np.zeros((data[tar].nunique(), data[can].nunique(), len(pc_all)))
        for key_pc, value_pc in data.groupby(pc):
            # N_kij[key_pc] = {k: v[can].value_counts().to_dict() for k, v in value_pc.groupby(tar)}
            N_kij_dict[key_pc] = {k: v[can].value_counts() for k, v in value_pc.groupby(tar)}
        for key_pc in pc_all:
            if len(key_pc) == 1:
                if key_pc[0] not in N_kij_dict:
                    N_kij_dict[key_pc[0]] = {}
                for key_tar in data[tar].unique():
                    if key_tar not in N_kij_dict[key_pc[0]]:
                        N_kij_dict[key_pc[0]][key_tar] = pd.Series([])
                    for key_can in data[can].unique():
                        if key_can not in N_kij_dict[key_pc[0]][key_tar]:
                            N_kij_dict[key_pc[0]][key_tar][key_can] = 0
                    N_kij_dict[key_pc[0]][key_tar] = N_kij_dict[key_pc[0]][key_tar].sort_index()
                N_kij[:, :, pc_all.index(key_pc)] = np.array([list(N_kij_dict[key_pc[0]][i].values) for i in N_kij_dict[key_pc[0]]])

            else:
                if key_pc not in N_kij_dict:
                    N_kij_dict[key_pc] = {}
                for key_tar in data[tar].unique():
                    if key_tar not in N_kij_dict[key_pc]:
                        N_kij_dict[key_pc][key_tar] = pd.Series([])
                    for key_can in data[can].unique():
                        if key_can not in N_kij_dict[key_pc][key_tar]:
                            N_kij_dict[key_pc][key_tar][key_can] = 0
                    N_kij_dict[key_pc][key_tar] = N_kij_dict[key_pc][key_tar].sort_index()
                N_kij[:, :, pc_all.index(key_pc)] = np.array([list(N_kij_dict[key_pc][i].values) for i in N_kij_dict[key_pc]])

    else:
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        # N_kij = {k: v[can].value_counts().to_dict() for k, v in data.groupby(tar)}
        N_kij_dict = {k: v[can].value_counts() for k, v in data.groupby(tar)}
        for key_tar in data[tar].unique():
            if key_tar not in N_kij_dict:
                N_kij_dict[key_tar] = pd.Series([])
            for key_can in data[can].unique():
                if key_can not in N_kij_dict[key_tar]:
                    N_kij_dict[key_tar][key_can] = 0
            N_kij_dict[key_tar] = N_kij_dict[key_tar].sort_index()
        N_kij = np.array([list(N_kij_dict[i].values) for i in N_kij_dict])

    return N_kij, dof

# contingency table calculator with compensation
def count_number_compensation(tar, pc, can, data, epsilon = 0.1):
    values = data[tar].unique()
    len_values = len(values)
    noise_tar = epsilon / (len_values - 1) * np.ones((len_values, len_values)) + (1 - epsilon * len_values / (len_values - 1)) * np.identity(len_values)

    values = data[can].unique()
    len_values = len(values)
    noise_can = epsilon / (len_values - 1) * np.ones((len_values, len_values)) + (1 - epsilon * len_values / (len_values - 1)) * np.identity(len_values)

    if pc:
        N_kij_dict = {}
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        pc_all = []
        noise_pc_all = {}
        for pc_var in pc:
            dof = dof * data[pc_var].nunique()
            pc_all.append(data[pc_var].unique())
            values = data[pc_var].unique()
            len_values = len(values)
            noise_pc_all[pc_var] = epsilon / (len_values - 1) * np.ones((len_values, len_values)) + (1 - epsilon * len_values / (len_values - 1)) * np.identity(len_values)
        pc_all = list(itertools.product(*pc_all))
        N_kij = np.zeros((data[tar].nunique(), data[can].nunique(), len(pc_all)))
        N_kij_compensation = np.zeros((data[tar].nunique(), data[can].nunique(), len(pc_all)))

        noise_pc = np.zeros((len(pc_all), len(pc_all)))

        for key_pc, value_pc in data.groupby(pc):
            N_kij_dict[key_pc] = {k: v[can].value_counts() for k, v in value_pc.groupby(tar)}
        for key_pc in pc_all:
            if len(key_pc) == 1:
                if key_pc[0] not in N_kij_dict:
                    N_kij_dict[key_pc[0]] = {}
                for key_tar in data[tar].unique():
                    if key_tar not in N_kij_dict[key_pc[0]]:
                        N_kij_dict[key_pc[0]][key_tar] = pd.Series([])
                    for key_can in data[can].unique():
                        if key_can not in N_kij_dict[key_pc[0]][key_tar]:
                            N_kij_dict[key_pc[0]][key_tar][key_can] = 0
                    N_kij_dict[key_pc[0]][key_tar] = N_kij_dict[key_pc[0]][key_tar].sort_index()
                N_kij[:, :, pc_all.index(key_pc)] = np.array([list(N_kij_dict[key_pc[0]][i].values) for i in N_kij_dict[key_pc[0]]])
                N_kij_compensation[:, :, pc_all.index(key_pc)] = (np.linalg.inv(noise_tar).dot(N_kij[:, :, pc_all.index(key_pc)])).dot(np.linalg.inv(noise_can))

                noise_pc[pc_all.index(key_pc)] = noise_pc_all[pc[0]][pc_all.index(key_pc)]
            else:
                if key_pc not in N_kij_dict:
                    N_kij_dict[key_pc] = {}
                for key_tar in data[tar].unique():
                    if key_tar not in N_kij_dict[key_pc]:
                        N_kij_dict[key_pc][key_tar] = pd.Series([])
                    for key_can in data[can].unique():
                        if key_can not in N_kij_dict[key_pc][key_tar]:
                            N_kij_dict[key_pc][key_tar][key_can] = 0
                    N_kij_dict[key_pc][key_tar] = N_kij_dict[key_pc][key_tar].sort_index()
                N_kij[:, :, pc_all.index(key_pc)] = np.array([list(N_kij_dict[key_pc][i].values) for i in N_kij_dict[key_pc]])
                N_kij_compensation[:, :, pc_all.index(key_pc)] = (np.linalg.inv(noise_tar).dot(N_kij[:, :, pc_all.index(key_pc)])).dot(np.linalg.inv(noise_can))

                noise_pc_sin = noise_pc_all[pc[0]][:, np.where(data[pc[0]].unique() == key_pc[0])]
                noise_pc_sin = np.asarray(noise_pc_sin).reshape((noise_pc_sin.size, 1))
                for k in range(len(key_pc) - 1):
                    noise_pc_temp = noise_pc_all[pc[k + 1]][np.where(data[pc[k + 1]].unique() == key_pc[k + 1]), :]
                    noise_pc_temp = np.asarray(noise_pc_temp).reshape((1, noise_pc_temp.size))
                    noise_pc_sin = noise_pc_sin.dot(noise_pc_temp)
                    noise_pc_sin = noise_pc_sin.reshape((noise_pc_sin.size, 1))

                noise_pc[:, pc_all.index(key_pc)] = noise_pc_sin.reshape((noise_pc_sin.size))

        for i in range(N_kij_compensation.shape[0]):
            N_kij_compensation[i, :, :] = N_kij_compensation[i, :, :].dot(np.linalg.inv(noise_pc))
    else:
        dof = (data[tar].nunique() - 1) * (data[can].nunique() - 1)
        N_kij_dict = {k: v[can].value_counts() for k, v in data.groupby(tar)}
        for key_tar in data[tar].unique():
            if key_tar not in N_kij_dict:
                N_kij_dict[key_tar] = pd.Series([])
            for key_can in data[can].unique():
                if key_can not in N_kij_dict[key_tar]:
                    N_kij_dict[key_tar][key_can] = 0
            N_kij_dict[key_tar] = N_kij_dict[key_tar].sort_index()
        N_kij = np.array([list(N_kij_dict[i].values) for i in N_kij_dict])

        N_kij_compensation = (np.linalg.inv(noise_tar).dot(N_kij)).dot(np.linalg.inv(noise_can))
    return N_kij_compensation, dof

# p-value calculator (based on G-test)
def p_value_calculator(N_kij, pc, dof):
    if pc:
        G = 0
        for k in range(N_kij.shape[2]):
            N_div = np.ones(N_kij.shape[0:2])
            N_div = np.multiply(N_div, N_kij[:, :, k].sum(axis=0))
            N_div = np.multiply(N_div, N_kij[:, :, k].sum(axis=1).reshape(N_kij[:, :, k].shape[0], 1))

            np.seterr(all='ignore')
            G = G + np.nansum(np.multiply(2 * N_kij[:, :, k], np.log(np.divide(N_kij[:, :, k] * N_kij[:, :, k].sum(), N_div))))
    else:
        N_div = np.ones(N_kij.shape)
        N_div = np.multiply(N_div, N_kij.sum(axis=0))
        N_div = np.multiply(N_div, N_kij.sum(axis=1).reshape(N_kij.shape[0], 1))

        np.seterr(all='ignore')
        G = np.nansum(np.multiply(2 * N_kij, np.log(np.divide(N_kij * N_kij.sum(), N_div))))
    p_value = chi2.sf(G, dof)
    return p_value

# # p-value calculator (based on G-test)
# def p_value_calculator(N_kij, pc, dof):
#     G = 0
#     if pc:
#         for key_pc in N_kij:
#             N_kj = {}
#             for key_tar in N_kij[key_pc]:
#                 N_kj = {k : N_kj.get(k, 0) + N_kij[key_pc][key_tar].get(k, 0) for k in set(N_kj) | set(N_kij[key_pc][key_tar])}
#             for key_tar in N_kij[key_pc]:
#                 for key_can in N_kij[key_pc][key_tar]:
#                     G = G + 2 * N_kij[key_pc][key_tar][key_can] * np.log(N_kij[key_pc][key_tar][key_can] * sum(N_kj.values()) /
#                         sum(N_kij[key_pc][key_tar].values()) / N_kj[key_can])
#     else:
#         N_kj = {}
#         for key_tar in N_kij:
#             N_kj = {k : N_kj.get(k, 0) + N_kij[key_tar].get(k, 0) for k in set(N_kj) | set(N_kij[key_tar])}
#         for key_tar in N_kij:
#             for key_can in N_kij[key_tar]:
#                 G = G + 2 * N_kij[key_tar][key_can] * np.log(N_kij[key_tar][key_can] * sum(N_kj.values()) /sum(N_kij[key_tar].values()) / N_kj[key_can])
#
#     p_value = chi2.sf(G, dof)
#     return p_value

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