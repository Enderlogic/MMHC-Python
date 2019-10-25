import numpy as np
from scipy.special import gammaln
import copy

# def hc(data, pc, state, method):
#     # this method compare two candidate graph by computing the nodes with different parents only
#     # hill climbing algorithm for discovering the edges
#     gra = {}
#     gra_temp = {}
#     for node in state:
#         gra[node] = []
#         gra_temp[node] = []
#     # sco_temp = score(gra, data, state, 'bdeu')
#
#     sco_diff_temp = 1
#     while sco_diff_temp > 0:
#         edge_candidate = []
#         sco_diff_temp = 0
#         gra_temp = copy.deepcopy(gra)
#         # attempting to add new edges
#         for var, pc_set in pc.items():
#             for pc_sin in pc_set:
#                 gra_temp[pc_sin].append(var)
#                 s_diff_temp = score_diff(gra_temp, gra, data, state, method)
#                 if s_diff_temp > sco_diff_temp:
#                     sco_diff_temp = s_diff_temp
#                     edge_candidate = [var, pc_sin, 'a']
#                 gra_temp[pc_sin].remove(var)
#
#         for var, par_set in gra_temp.items():
#             for par_sin in par_set:
#                 # attempting to reverse existing edges
#                 gra_temp[par_sin].append(var)
#                 gra_temp[var].remove(par_sin)
#                 s_diff_temp = score_diff(gra_temp, gra, data, state, method)
#                 if s_diff_temp > sco_diff_temp:
#                     sco_diff_temp = s_diff_temp
#                     edge_candidate = [var, par_sin, 'r']
#                 gra_temp[par_sin].remove(var)
#                 gra_temp[var].append(par_sin)
#                 # attempting to delete existing edges
#                 gra_temp[var].remove(par_sin)
#                 s_diff_temp = score_diff(gra_temp, gra, data, state, method)
#                 if s_diff_temp > sco_diff_temp:
#                     sco_diff_temp = s_diff_temp
#                     edge_candidate = [var, par_sin, 'd']
#                 gra_temp[var].append(par_sin)
#
#         if edge_candidate:
#             if edge_candidate[-1] == 'a':
#                 gra[edge_candidate[1]].append(edge_candidate[0])
#                 pc[edge_candidate[1]].remove(edge_candidate[0])
#                 pc[edge_candidate[0]].remove(edge_candidate[1])
#             elif edge_candidate[-1] == 'r':
#                 gra[edge_candidate[1]].remove(edge_candidate[0])
#                 gra[edge_candidate[0]].remove(edge_candidate[1])
#             elif edge_candidate[-1] == 'd':
#                 gra[edge_candidate[1]].remove(edge_candidate[0])
#                 pc[edge_candidate[1]].append(edge_candidate[0])
#                 pc[edge_candidate[0]].append(edge_candidate[1])
#     return gra

def score(gra, data, state, method):
    if method == 'bdeu':
        sco = score_bdeu(gra, data, state)
    return sco

def score_diff(gra1, gra2, data, state, method):
    if method == 'bdeu':
        sco_diff = score_bdeu_diff(gra1, gra2, data, state)
    return sco_diff

def score_bdeu(gra, data, state):
    num = {}
    size_par = {}
    size_var = {}
    for var in state:
        size_var[var] = len(state[var])
        size_par[var] = 1
        if gra[var]:
            for par in gra[var]:
                size_par[var] = size_par[var] * len(state[par])
        num[var] = np.zeros((size_var[var], size_par[var]))

    for i in range(data.shape[0]):
        for var in state:
            ind_par = 0
            if gra[var]:
                for par in gra[var]:
                    ind_par = ind_par * len(state[par]) + state[par].index(data[par][i])
            ind_var = state[var].index(data[var][i])
            num[var][ind_var][ind_par] = num[var][ind_var][ind_par] + 1
    bdeu = 0
    iss = 1
    for var in state:
        for col in range(size_par[var]):
            bdeu = bdeu + gammaln(iss / size_par[var]) - gammaln(iss / size_par[var] + num[var].sum(axis = 0)[col])
            for row in range(size_var[var]):
                bdeu = bdeu + gammaln(iss / size_var[var] / size_par[var] + num[var][row][col]) - gammaln(iss / size_var[var] / size_par[var])

    return bdeu

def score_bdeu_diff(gra1, gra2, data ,state):
    num = {}
    num['gra1'] = {}
    num['gra2'] = {}
    size_par = {}
    size_par['gra1'] = {}
    size_par['gra2'] = {}
    size_var = {}
    for var in state:
        if not (gra1[var] == gra2[var]):
            size_var[var] = len(state[var])
            size_par['gra1'][var] = 1
            size_par['gra2'][var] = 1
            if gra1[var]:
                for par in gra1[var]:
                    size_par['gra1'][var] = size_par['gra1'][var] * len(state[par])
            if gra2[var]:
                for par in gra2[var]:
                    size_par['gra2'][var] = size_par['gra2'][var] * len(state[par])
            num['gra1'][var] = np.zeros((size_var[var], size_par['gra1'][var]))
            num['gra2'][var] = np.zeros((size_var[var], size_par['gra2'][var]))

    for i in range(data.shape[0]):
        for var in state:
            if not (gra1[var] == gra2[var]):
                ind_par1 = 0
                ind_par2 = 0
                if gra1[var]:
                    for par in gra1[var]:
                        ind_par1 = ind_par1 * len(state[par]) + state[par].index(data[par][i])
                if gra2[var]:
                    for par in gra2[var]:
                        ind_par2 = ind_par2 * len(state[par]) + state[par].index(data[par][i])
                ind_var = state[var].index(data[var][i])
                num['gra1'][var][ind_var][ind_par1] = num['gra1'][var][ind_var][ind_par1] + 1
                num['gra2'][var][ind_var][ind_par2] = num['gra2'][var][ind_var][ind_par2] + 1
    bdeu_diff = 0
    iss = 1
    for var in state:
        if not (gra1[var] == gra2[var]):
            for col in range(size_par['gra1'][var]):
                bdeu_diff = bdeu_diff + gammaln(iss / size_par['gra1'][var]) - gammaln(iss / size_par['gra1'][var] + num['gra1'][var].sum(axis = 0)[col])
                for row in range(size_var[var]):
                    bdeu_diff = bdeu_diff + gammaln(iss / size_var[var] / size_par['gra1'][var] + num['gra1'][var][row][col]) - gammaln(iss / size_var[var] / size_par['gra1'][var])
            for col in range(size_par['gra2'][var]):
                bdeu_diff = bdeu_diff - (gammaln(iss / size_par['gra2'][var]) - gammaln(iss / size_par['gra2'][var] + num['gra2'][var].sum(axis = 0)[col]))
                for row in range(size_var[var]):
                    bdeu_diff = bdeu_diff - (gammaln(iss / size_var[var] / size_par['gra2'][var] + num['gra2'][var][row][col]) - gammaln(iss / size_var[var] / size_par['gra2'][var]))

    return bdeu_diff

def hc(data, pc, state, method):
    # this method compute the score of two candidate graph and compare them with their value
    # hill climbing algorithm for discovering the edges
    sco = float("-inf")
    gra = {}
    gra_temp = {}
    for node in state:
        gra[node] = []
        gra_temp[node] = []
    sco_temp = score(gra, data, state, method)

    while sco_temp > sco:
        edge_candidate = []
        sco = sco_temp
        gra_temp = gra.copy()
        # attempting to add new edges
        for var, pc_set in pc.items():
            for pc_sin in pc_set:
                gra_temp[pc_sin].append(var)
                s_temp = score(gra_temp, data, state, method)
                if s_temp > sco_temp:
                    sco_temp = s_temp
                    edge_candidate = [var, pc_sin, 'a']
                gra_temp[pc_sin].remove(var)

        for var, par_set in gra_temp.items():
            for par_sin in par_set:
                # attempting to reverse existing edges
                gra_temp[par_sin].append(var)
                gra_temp[var].remove(par_sin)
                s_temp = score(gra_temp, data, state, method)
                if s_temp > sco_temp:
                    sco_temp = s_temp
                    edge_candidate = [var, par_sin, 'r']
                gra_temp[par_sin].remove(var)
                gra_temp[var].append(par_sin)
                # attempting to delete existing edges
                gra_temp[var].remove(par_sin)
                s_temp = score(gra_temp, data, state, method)
                if s_temp > sco_temp:
                    sco_temp = s_temp
                    edge_candidate = [var, par_sin, 'd']
                gra_temp[var].append(par_sin)

        if edge_candidate:
            if edge_candidate[-1] == 'a':
                gra[edge_candidate[1]].append(edge_candidate[0])
                pc[edge_candidate[1]].remove(edge_candidate[0])
                pc[edge_candidate[0]].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'r':
                gra[edge_candidate[1]].remove(edge_candidate[0])
                gra[edge_candidate[0]].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                gra[edge_candidate[1]].remove(edge_candidate[0])
                pc[edge_candidate[1]].append(edge_candidate[0])
                pc[edge_candidate[0]].append(edge_candidate[1])
    return gra