import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve


def show_pr_plot(precision, recall, title, color='b'):
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color=color, alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title(title)
    plt.grid(True)
    plt.show()
    # Wait for user input to close the plot
    while True:
        user_input = input("Press 'q' to close the plot: ")
        if user_input.lower() == 'q':
            break
    plt.close()


if __name__ == '__main__':
    hlnet_test = np.load("np_predictions/test/off_hl-net.npy")
    hlnet_test = np.repeat(hlnet_test, 16)

    cma_test = np.load("np_predictions/test/off_cma.npy")
    cma_test = np.repeat(cma_test, 16)

    hypervd_test = np.load("np_predictions/test/off_hypervd.npy")
    hypervd_test = np.repeat(hypervd_test, 16)

    macilsd_test = np.load("np_predictions/test/off_macil_sd.npy")
    macilsd_test = np.repeat(macilsd_test, 16)

    msbt_test = np.load("np_predictions/test/off_MSBT_test.npy")

    # Metamodel training set
    metamodel_train_pred = np.load("np_predictions/train/train_metamodel.npy")
    metamodel_test_pred = np.load("np_predictions/test/metamodel.npy")

    gt_train = np.load("list/newsplit/train/gt.npy")
    gt_test = np.load("list/newsplit/test/gt.npy")

    train_precision, train_recall, train_th = precision_recall_curve(list(gt_train), metamodel_train_pred)
    show_pr_plot(train_precision, train_recall, 'Training set - Precision-Recall Curve')

    # Metamodel test set
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), metamodel_test_pred)
    show_pr_plot(test_precision, test_recall, 'Test set - Precision-Recall Curve', 'r')


    #HL-NET
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), hlnet_test)
    show_pr_plot(test_precision, test_recall, 'HLnet Test set - Precision-Recall Curve', 'c')

    #CMA
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), cma_test)
    show_pr_plot(test_precision, test_recall, 'Cma Test set - Precision-Recall Curve', '#000000')

    #HYPERVD
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), hypervd_test)
    show_pr_plot(test_precision, test_recall, 'HyperVD Test set - Precision-Recall Curve', '#FF00FF')

    #MACIL-SD
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), macilsd_test)
    show_pr_plot(test_precision, test_recall, 'MacilSD Test set - Precision-Recall Curve', '#00FF00')

    # MSBT
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), msbt_test)
    show_pr_plot(test_precision, test_recall, 'MSBT Test set - Precision-Recall Curve', '#FF00FF')