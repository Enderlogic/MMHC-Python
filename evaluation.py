# SHD method
def compare(target, current):
    # input:
    # target: true DAG
    # current: learned DAG
    # output:
    # tp: true positive (edges appear in both target and current)
    # fp: false positive (edges appear in current but not in target)
    # fn: false negative (edges appear in target but not in current)

    compare_dict = {}
    tp = 0
    fp = 0
    fn = 0
    for var, parents in target.items():
        for par in parents:
            if par in current[var]:
                tp = tp + 1
            else:
                fn = fn + 1
    for var, parents in current.items():
        for par in parents:
            if par not in target[var]:
                fp = fp + 1

    compare_dict['tp'] = tp
    compare_dict['fp'] = fp
    compare_dict['fn'] = fn
    return compare_dict