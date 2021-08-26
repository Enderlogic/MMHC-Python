from rpy2.robjects.packages import importr

from lib.accessory import local_score, to_bnlearn

base, bnlearn = importr('base'), importr('bnlearn')


def check_cycle(vi, vj, dag):
    # whether adding or orientating edge vi->vj would cause cycle. In other words, this function check whether there is a direct path from vj to vi except the possible edge vi<-vj
    underchecked = [x for x in dag[vi]['par'] if x != vj]
    checked = []
    cyc_flag = False
    while underchecked:
        if cyc_flag:
            break
        underchecked_copy = list(underchecked)
        for vk in underchecked_copy:
            if dag[vk]['par']:
                if vj in dag[vk]['par']:
                    cyc_flag = True
                    break
                else:
                    for key in dag[vk]['par']:
                        if key not in checked + underchecked:
                            underchecked.append(key)
            underchecked.remove(vk)
            checked.append(vk)
    return cyc_flag


def hc(data, arities, varnames, pc=None, score='default'):
    '''
    :param data: the training data used for learn BN (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param pc: the candidate parents and children set for each variable
    :param score: score function, including:
                   bic (Bayesian Information Criterion for discrete variable)
                   bic_g (Bayesian Information Criterion for continuous variable)

    :return: the learned BN (bnlearn format)
    '''
    if score == 'default':
        score = 'bic_g' if arities is None else 'bic'
    # initialize the candidate parents-set for each variable
    candidate = {}
    dag = {}
    cache = {}
    for var in varnames:
        if pc is None:
            candidate[var] = list(varnames)
            candidate[var].remove(var)
        else:
            candidate[var] = list(pc[var])
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        cache[var] = {}
        cache[var][tuple([])] = local_score(data, arities, [varnames.index(var)], score)
    diff = 1
    while diff > 0:
        diff = 0
        edge_candidate = []
        for vi in varnames:
            # attempt to add edges vi->vj
            for vj in candidate[vi]:
                cyc_flag = check_cycle(vi, vj, dag)
                if not cyc_flag:
                    par_sea = tuple(sorted(dag[vj]['par'] + [vi]))
                    if par_sea not in cache[vj]:
                        cols = [varnames.index(x) for x in (vj, ) + par_sea]
                        cache[vj][par_sea] = local_score(data, arities, cols, score)
                    diff_temp = cache[vj][par_sea] - cache[vj][tuple(dag[vj]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [vi, vj, 'a']
            for par_vi in dag[vi]['par']:
                # attempt to reverse edges from vi<-par_vi to vi->par_vi
                cyc_flag = check_cycle(vi, par_vi, dag)
                if not cyc_flag:
                    par_sea_par_vi = tuple(sorted(dag[par_vi]['par'] + [vi]))
                    if par_sea_par_vi not in cache[par_vi]:
                        cols = [varnames.index(x) for x in (par_vi, ) + par_sea_par_vi]
                        cache[par_vi][par_sea_par_vi] = local_score(data, arities, cols, score)
                    par_sea_vi = tuple([x for x in dag[vi]['par'] if x != par_vi])
                    if par_sea_vi not in cache[vi]:
                        cols = [varnames.index(x) for x in (vi, ) + par_sea_vi]
                        cache[vi][par_sea_vi] = local_score(data, arities, cols, score)
                    diff_temp = cache[par_vi][par_sea_par_vi] + cache[vi][par_sea_vi] - cache[par_vi][
                        tuple(dag[par_vi]['par'])] - cache[vi][tuple(dag[vi]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [vi, par_vi, 'r']
                # attempt to delete edges vi<-par_vi
                par_sea = tuple([x for x in dag[vi]['par'] if x != par_vi])
                if par_sea not in cache[vi]:
                    cols = [varnames.index(x) for x in (vi, ) + par_sea]
                    cache[vi][par_sea] = local_score(data, arities, cols, score)
                diff_temp = cache[vi][par_sea] - cache[vi][tuple(dag[vi]['par'])]
                if diff_temp - diff > 1e-10:
                    diff = diff_temp
                    edge_candidate = [par_vi, vi, 'd']
        if edge_candidate:
            if edge_candidate[-1] == 'a':
                dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                candidate[edge_candidate[0]].remove(edge_candidate[1])
                candidate[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                dag[edge_candidate[1]]['par'].remove(edge_candidate[0])
                candidate[edge_candidate[0]].append(edge_candidate[1])
                candidate[edge_candidate[1]].append(edge_candidate[0])
    dag = bnlearn.model2network(to_bnlearn(dag))
    return dag
