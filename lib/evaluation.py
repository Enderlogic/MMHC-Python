# evaluation methods
from rpy2.robjects.packages import importr
base, bnlearn = importr('base'), importr('bnlearn')

# compute the F1 score of a learned graph given true graph
def f1(dag_true, dag_learned):
    '''
    :param dag_true: true DAG
    :param dag_learned: learned DAG
    :return: the F1 score of learned DAG
    '''
    compare = bnlearn.compare(bnlearn.cpdag(dag_true), bnlearn.cpdag(dag_learned))
    return compare[0][0] * 2 / (compare[0][0] * 2 + compare[1][0] + compare[2][0])