import numpy as np
from scipy.special import gammaln
import copy

def score_diff(gra1, gra2, data, score_function):
    if score_function == 'bdeu':
        sco_diff = score_bdeu_diff(gra1, gra2, data)
    elif score_function == 'bic':
        sco_diff = score_bic_diff(gra1, gra2, data)
    return sco_diff

# compare the bdeu score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bdeu_diff(gra1, gra2, data):
    iss = 1
    bdeu_diff = 0
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            contingency_table1 = {}
            contingency_table2 = {}
            q1 = 1
            q2 = 1
            r = len(data[tar].unique())

            if gra1[tar]:
                contingency_table1['N_jk'] = {}
                for key_par, value_par in data.groupby(gra1[tar]):
                    contingency_table1['N_jk'][key_par] = {k:v.shape[0] for k, v in value_par.groupby(tar)}

                for par_var in gra1[tar]:
                    q1 = q1 * len(data[par_var].unique())
                alp1_j = iss / q1
                alp1_jk = iss / q1 / r

                for key_par in contingency_table1['N_jk']:
                    for key_tar in contingency_table1['N_jk'][key_par]:
                        bdeu_diff = bdeu_diff - gammaln(alp1_jk + contingency_table1['N_jk'][key_par][key_tar]) + gammaln(alp1_jk)
                    bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + sum(contingency_table1['N_jk'][key_par].values()))
            else:
                alp1_j = iss
                alp1_jk = iss / r
                contingency_table1['N_k'] =  {k:v.shape[0] for k, v in data.groupby(tar)}
                bdeu_diff = bdeu_diff - gammaln(alp1_j) + gammaln(alp1_j + data[tar].shape[0])
                for key_tar in contingency_table1['N_k']:
                    bdeu_diff = bdeu_diff - gammaln(alp1_jk + contingency_table1['N_k'][key_tar]) + gammaln(alp1_jk)

            if gra2[tar]:
                contingency_table2['N_jk'] = {}
                for key_par, value_par in data.groupby(gra2[tar]):
                    contingency_table2['N_jk'][key_par] = {k:v.shape[0] for k, v in value_par.groupby(tar)}

                for par_var in gra2[tar]:
                    q2 = q2 * len(data[par_var].unique())

                alp2_j = iss / q2
                alp2_jk = iss / q2 / r

                for key_par in contingency_table2['N_jk']:
                    for key_tar in contingency_table2['N_jk'][key_par]:
                        bdeu_diff = bdeu_diff + gammaln(alp2_jk + contingency_table2['N_jk'][key_par][key_tar]) - gammaln(alp2_jk)
                    bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + sum(contingency_table2['N_jk'][key_par].values()))
            else:
                alp2_j = iss
                alp2_jk = iss / r
                contingency_table2['N_k'] = {k: v.shape[0] for k, v in data.groupby(tar)}
                bdeu_diff = bdeu_diff + gammaln(alp2_j) - gammaln(alp2_j + data[tar].shape[0])
                for key_tar in contingency_table2['N_k']:
                    bdeu_diff = bdeu_diff + gammaln(alp2_jk + contingency_table2['N_k'][key_tar]) - gammaln(alp2_jk)
    return bdeu_diff

# compare the bic score between gra1 and gra2, a positive value means bdeu(gra2) > bdeu(gra1)
def score_bic_diff(gra1, gra2, data):
    bic_diff = 0
    for tar in data:
        if set(gra1[tar]) != set(gra2[tar]):
            contingency_table1 = {}
            contingency_table2 = {}
            q1 = 1
            q2 = 1
            r = len(data[tar].unique())

            if gra1[tar]:
                contingency_table1['N_jk'] = {}
                for key_par, value_par in data.groupby(gra1[tar]):
                    contingency_table1['N_jk'][key_par] = {k: v.shape[0] for k, v in value_par.groupby(tar)}

                for par_var in gra1[tar]:
                    q1 = q1 * len(data[par_var].unique())

                for key_par in contingency_table1['N_jk']:
                    for key_tar in contingency_table1['N_jk'][key_par]:
                        bic_diff = bic_diff - contingency_table1['N_jk'][key_par][key_tar] * np.log(contingency_table1['N_jk'][key_par][key_tar] / sum(contingency_table1['N_jk'][key_par].values()))

            else:
                contingency_table1['N_k'] = {k: v.shape[0] for k, v in data.groupby(tar)}
                for key_tar in contingency_table1['N_k']:
                    bic_diff = bic_diff - contingency_table1['N_k'][key_tar] * np.log(contingency_table1['N_k'][key_tar] / data.shape[0])

            if gra2[tar]:
                contingency_table2['N_jk'] = {}
                for key_par, value_par in data.groupby(gra2[tar]):
                    contingency_table2['N_jk'][key_par] = {k: v.shape[0] for k, v in value_par.groupby(tar)}

                for par_var in gra2[tar]:
                    q2 = q2 * len(data[par_var].unique())

                for key_par in contingency_table2['N_jk']:
                    for key_tar in contingency_table2['N_jk'][key_par]:
                        bic_diff = bic_diff + contingency_table2['N_jk'][key_par][key_tar] * np.log(contingency_table2['N_jk'][key_par][key_tar] / sum(contingency_table2['N_jk'][key_par].values()))

            else:
                contingency_table2['N_k'] = {k: v.shape[0] for k, v in data.groupby(tar)}
                for key_tar in contingency_table2['N_k']:
                    bic_diff = bic_diff + contingency_table2['N_k'][key_tar] * np.log(contingency_table2['N_k'][key_tar] / data.shape[0])

            bic_diff = bic_diff + np.log(data.shape[0]) / 2 * (r - 1) * (q1 - q2)
    return bic_diff

# hill climbing algorithm
def hc(data, pc, score_function):

    gra = {}
    gra_temp = {}
    for node in data:
        gra[node] = []
        gra_temp[node] = []

    diff = 1

    # attempt to find better graph until no difference could make
    while diff > 1e-10:

        diff = 0
        edge_candidate = []
        gra_temp = copy.deepcopy(gra)

        for tar in data:
            # attempt to add edges
            for pc_var in pc[tar]:
                gra_temp[tar].append(pc_var)

                score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                if score_diff_temp > diff:
                    diff = score_diff_temp
                    edge_candidate = [tar, pc_var, 'a']

                gra_temp[tar].remove(pc_var)

            for par_var in gra[tar]:
                # attempt to reverse edges
                gra_temp[par_var].append(tar)
                gra_temp[tar].remove(par_var)

                score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                if score_diff_temp > diff:
                    diff = score_diff_temp
                    edge_candidate = [tar, par_var, 'r']

                gra_temp[par_var].remove(tar)

                # attempt to delete edges
                score_diff_temp = score_diff(gra, gra_temp, data, score_function)
                if score_diff_temp > diff:
                    diff = score_diff_temp
                    edge_candidate = [tar, par_var, 'd']

                gra_temp[tar].append(par_var)

        # print(diff)
        # print(edge_candidate)

        if edge_candidate:
            if edge_candidate[-1] == 'a':
                gra[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                gra[edge_candidate[1]].append(edge_candidate[0])
                gra[edge_candidate[0]].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                gra[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[1]].append(edge_candidate[0])

    return gra