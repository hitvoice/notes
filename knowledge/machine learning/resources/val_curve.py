import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve


def plot_curve(model, X, y, param_name, param_range, nfolds=3):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=-1, cv=nfolds)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

    def print_cv(param, test_score):
            print(('%.2e ('+ '|'.join(['%.2f']*len(test_score)) + ') %.3f') % 
                  tuple([param]+list(test_score)+[np.mean(test_score)]))

    for param, test_score in zip(param_range, test_scores):
        print_cv(param, test_score)