from .general_utils import *
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


# def extract_MIA_features(model, dataloader):
#     feas, labels = extract_features(model, dataloader, {'classifier.3': 'hidden'})


def plot_prec_recall(clf, test_data, save=False):
    pred_probs = []
    for data in test_data:
        feas, labels, name = data
        pred_prob = clf.predict_proba(feas)
        pred_probs.append(pred_prob)

    for pred_prob, data in zip(pred_probs, test_data):
        feas, labels, name = data
        precision, recall, thresholds = metrics.precision_recall_curve(labels, pred_prob[:, 1])
        auc = metrics.auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} - AUC: {auc:.04f}')

    plt.grid(True)
    plt.legend()
    plt.show()

    for pred_prob, data in zip(pred_probs, test_data):
        feas, labels, name = data
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_prob[:, 1])
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} - AUC: {auc:.04f}')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.show()


    for pred_prob, data in zip(pred_probs, test_data):
        feas, labels, name = data
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_prob[:, 1])
        if save:
            np.save(f'exp/results/ROC_{name}.npy', {'fpr': fpr, 'tpr': tpr})
        plt.plot(np.log(fpr + 1e-10), np.log(tpr + 1e-10), label=f'{name}')

    plt.xlim([-5, 0])
    plt.ylim([-5, 0])
    plt.grid(True)
    plt.legend()
    plt.show()
    

    