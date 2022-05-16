from itertools import combinations

def pred_tree(clf, coalition, row, node=0):
    """
        The following function computes the characteristic function by
        relying on the tree structure
        clf : ml model we used for predictions
        coalition : the features that constitute the coalition
        row : the observation for which we'll compute the banzhaf values
    """
    left_node = clf.tree_.children_left[node]
    right_node = clf.tree_.children_right[node]
    is_leaf = left_node == right_node
    if is_leaf:
        return clf.tree_.value[node].squeeze()
    feature = row.index[clf.tree_.feature[node]]
    if feature in coalition:
        if row.loc[feature] <= clf.tree_.threshold[node]:
            # go left
            return pred_tree(clf, coalition, row, node=left_node)
        # go right
        return pred_tree(clf, coalition, row, node=right_node)
    # take weighted average of left and right
    wl = clf.tree_.n_node_samples[left_node] / clf.tree_.n_node_samples[node]
    wr = clf.tree_.n_node_samples[right_node] / clf.tree_.n_node_samples[node]
    value = wl * pred_tree(clf, coalition, row, node=left_node)
    value += wr * pred_tree(clf, coalition, row, node=right_node)
    return value

def make_value_function(clf, row, col):
    """
        Returns a function that computes the marginal gain of a feature
        clf : ml model we used for predictions
        row : observation
        col : column, that we want to determine its importanc
    """
    def value(c):
        marginal_gain = pred_tree(clf, c + [col], row) - pred_tree(clf, c, row)
        return marginal_gain
    return value

def make_coalitions(row, col):
    """
        The following function generates all possible coalitions of features exluding "col"
        row : the observation for which we cant to compute banzhaf values
        col : the col we exclude in the coalitions
    """
    rest = [x for x in row.index if x != col]
    for i in range(len(rest) + 1):
        for x in combinations(rest, i):
            yield list(x)

def compute_banzhaf(clf, row, col):
    """
        The following function computes the banzhaf values for a given observation "row", using the formula explicited in
        our report
        clf, row, col, df as described earlier
    """
    v = make_value_function(clf, row, col)
    return sum([v(coal) / (2 ** (len(row) - 1)) for coal in make_coalitions(row, col)])