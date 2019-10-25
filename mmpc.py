import numpy as np
from scipy.stats.distributions import chi2
import itertools
from collections import Counter

def mmpc_forward(data, state):

    #forward phase of MMPC
    pc_con = {}
    for tar in data:
        # check independence between two variables
        pc_con[tar], pc_rest = independence_test_binary(tar, data, state)

        if pc_rest:
            # check conditional independence conditioning on one variable
            p = independence_test_ternary(tar, pc_con[tar], pc_rest, data, state)
            p_store = p.copy()
            if p:
                pc_con[tar].append(min(p, key = p.get))
                p.pop(min(p, key=p.get))
            pc_rest = []
            while p:
                pc_rest.append(min(p, key = p.get))
                p.pop(min(p, key=p.get))
            # check conditional independence conditioning on two or more variables
            while pc_rest:
                for r in range(len(pc_con[tar])):
                    for subset in itertools.combinations(pc_con[tar], r):
                        pc_con_temp = list(subset)
                        pc_con_temp.append(pc_con[tar][-1])
                        p_temp = independence_test_ternary(tar, pc_con_temp, pc_rest, data, state)
                        pc_rest = []
                        if not p_temp:
                            break
                        p_store_temp = {}
                        for key in p_store:
                            if key in p_temp.keys():
                                p_store_temp[key] = p_store[key]
                        p_store = Counter(p_store_temp) + Counter(p_temp)
                        p_store_temp = p_store.copy()
                        while p_store_temp:
                            pc_rest.append(min(p_store_temp, key = p_store_temp.get))
                            p_store_temp.pop(min(p_store_temp, key = p_store_temp.get))
                    if not pc_rest:
                        break
                if p:
                    pc_con[tar].append(min(p, key=p.get))
                    p.pop(min(p, key=p.get))
                pc_rest = []
                while p:
                    pc_rest.append(min(p, key=p.get))
                    p.pop(min(p, key=p.get))
    return pc_con

def mmpc_backward(pc, data, state):

    #backward phase of MMPC
    pc_con = {}
    for tar in data:
        pc[tar].reverse()
        pc_con[tar] = [pc[tar][0]]
        pc_rest = pc[tar][1:]

        while pc_rest:
            p = independence_test_ternary(tar, pc_con[tar], pc_rest, data, state)

            if p:
                for che in pc_rest:
                    if che in p:
                        pc_con[tar].append(che)
                        p.pop(che)
                        break

            pc_rest_temp = []
            for che in pc_rest:
                if che in p:
                    pc_rest_temp.append(che)

            pc_rest = pc_rest_temp.copy()

    return pc_con

def asymmetry_test(pc):
    for key, value in pc.items():
        for var in value:
            if not (key in pc[var]):
                pc[key].remove(var)
    return pc

def independence_test_binary(tar, data, state):
    tar_size = len(state[tar])
    num_tar = np.zeros((tar_size, 1))
    num_che = {}
    num_co = {}

    for che in state.keys():
        if not (che is tar):
            che_size = len(state[che])
            num_che[che] = np.zeros((che_size, 1))
            num_co[che] = np.zeros((che_size, tar_size))

    for i in range(data.shape[0]):
        tar_state = state[tar].index(data[tar][i])
        num_tar[tar_state] = num_tar[tar_state] + 1

        for che in num_che.keys():
            che_state = state[che].index(data[che][i])
            num_che[che][che_state] = num_che[che][che_state] + 1
            num_co[che][che_state][tar_state] = num_co[che][che_state][tar_state] + 1

    p = {}
    for che in num_che.keys():
        G_temp = num_co[che] * np.log(num_co[che] * data.shape[0] / num_che[che].dot(num_tar.T))
        G_temp = G_temp.ravel()

        G = 2 * sum(G_temp[i] for i in range(len(G_temp)) if not np.isnan(G_temp[i]))
        # p_temp = 1 - stats.chi2.cdf(G, (len(state[tar]) - 1) * (len(state[che]) - 1))

        dof = (len(state[tar]) - 1) * (len(state[che]) - 1)

        p_temp = chi2.sf(G, dof)
        if p_temp < 0.05:
            p[che] = p_temp

    pc_con = []
    if p:
        pc_con.append(min(p, key = p.get))
        p.pop(min(p, key = p.get))
    pc_rest = []
    while p:
        pc_rest.append(min(p, key = p.get))
        p.pop(min(p, key = p.get))

    return pc_con, pc_rest

def independence_test_ternary(tar, pc_con, pc_rest, data, state):
    # initialise the counter
    num = {}
    size_con = 1
    for var_con in pc_con:
        size_con = size_con * len(state[var_con])
    num['con'] = np.zeros((size_con, 1))

    size_tar = len(state[tar])
    num[tar] = num['con'].dot(np.zeros((1, size_tar)))

    size_che = {}
    for che in pc_rest:
        che_set = {}
        size_che[che] = len(state[che])
        che_set['self'] = num['con'].dot(np.zeros((1, size_che[che])))
        che_set['co'] = np.expand_dims(che_set['self'], axis = -1).dot(np.zeros((1, size_tar)))

        num[che] = che_set

    # traverse the data set to count the S
    for i in range(data.shape[0]):
        con_state = 0
        for var_con in pc_con:
            con_state = con_state * len(state[var_con]) + state[var_con].index(data[var_con][i])

        num['con'][con_state] = num['con'][con_state] + 1

        tar_state = state[tar].index(data[tar][i])

        num[tar][con_state][tar_state] = num[tar][con_state][tar_state] + 1

        for che in pc_rest:
            che_state = state[che].index(data[che][i])
            num[che]['self'][con_state][che_state] = num[che]['self'][con_state][che_state] + 1
            num[che]['co'][con_state][che_state][tar_state] = num[che]['co'][con_state][che_state][tar_state] + 1

    #check the conditional independence by using G2 statistic
    p = {}
    for che in pc_rest:
        G = 0
        for i in range(size_con):
            G_ori = num[che]['co'][i] * np.log(num[che]['co'][i] * num['con'][i] / np.expand_dims(num[che]['self'][i], axis= -1).dot(np.expand_dims(num[tar][i], axis= 1).T))
            G_ori = G_ori.ravel()

            G = G + 2 * sum(G_ori[j] for j in range(len(G_ori)) if not np.isnan(G_ori[j]))

        # p_temp = 1 - stats.chi2.cdf(G, (np.count_nonzero(num[che]['co']) - np.count_nonzero(num[che]['self']) - np.count_nonzero(num[tar]) + np.count_nonzero(num['con'])))

        # dof = (size_tar - 1) * (size_che[che] - 1) * size_con
        dof = (np.count_nonzero(num[tar]) - np.count_nonzero(num['con'])) * (size_che[che] - 1)
        p_temp = chi2.sf(G, dof)
        if p_temp < 0.05:
            p[che] = p_temp

    return p