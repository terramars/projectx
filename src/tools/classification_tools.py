
from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, mean_squared_error, r2_score
import numpy as np

def mse_score(y_true,y_pred):
    labels = [(-1,12),(12,20),(20,35),(35,55),(55,1000)]
    weights = [1.5,1.5,.75,.75,1]
    labels = list((np.array(labels)-29.29)/17.8)
    scores = []
    for l in labels:
        idx = [i for i in range(len(y_true)) if y_true[i]<l[1] and y_true[i]>=l[0]]
        tmp_y_true = [y_true[i] for i in idx]
        tmp_y_pred = [y_pred[i] for i in idx]
        scores.append(np.sqrt(mean_squared_error(list(np.array(tmp_y_true)*17.8+29.29), list(np.array(tmp_y_pred)*17.8+29.29))))
    print 'mses',scores
    return 1.0 / (1e-6+sum([scores[i]*weights[i] for i in range(len(weights))])/len(scores))

class ParameterOptimizer(object):
    n_samples = None
    X = None
    y = None
    mode = None
    
    def __init__(self,data,labels,mode='precision'):
        self.n_samples = len(data)
        self.X = data
        self.y = labels
        self.mode = mode
    
    def optimize_parameters(self,target_class,tuned_parameters,**kwargs):
        if self.mode == 'precision':
            score_func = precision_score
        elif self.mode == 'mse':
            score_func = mse_score
        else:
            score_func = r2_score
        print "# Tuning hyper-parameters for %s" % self.mode
        print
        clf = GridSearchCV(target_class(**kwargs), tuned_parameters, score_func=score_func, cv=5, n_jobs = 1, refit = True, verbose = 2)
        clf.fit(self.X, self.y)
        print "Grid scores on development set:"
        print
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (
                mean_score, scores.std() / 2, params)
        print
        print "Best parameters set found on development set:"
        print
        print clf.best_estimator_
        return clf.best_estimator_
        
def evaluate_data(clf,y_test,X_test):
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)

def regression_report(y_true,y_pred):
    out = ''
    out += 'r2 score: ' + str(r2_score(y_true,y_pred))
    out += '\nrms score: ' + str(np.sqrt(mean_squared_error(y_true, y_pred)))
    return out

def unique_labels(*lists_of_labels):
    """Extract an ordered array of unique labels"""
    labels = set()
    for l in lists_of_labels:
        if hasattr(l, 'ravel'):
            l = l.ravel()
        labels |= set(l)
    return np.unique(sorted(labels))

def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report showing the main classification metrics

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets

    y_pred : array, shape = [n_samples]
        Estimated targets

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report

    target_names : list of strings
        Optional display names matching the labels (same order)

    Returns
    -------
    report : string
        Text summary of the precision, recall, f1-score for each class

    """
    from sklearn.metrics import precision_recall_fscore_support
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None)

    for i, _ in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        values += ["%d" % int(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["%0.2f" % float(v)]
    values += ['%d' % np.sum(s)]
    report += fmt % tuple(values)
    return report
    
    
