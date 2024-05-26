import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader

import option
from dataset import Dataset

if __name__ == '__main__':
    ## Training predictions
    hlnet_train = np.load("np_predictions/train/off_hl-net.npy")
    hlnet_train = np.repeat(hlnet_train, 16)

    cma_train = np.load("np_predictions/train/off_cma.npy")
    cma_train = np.repeat(cma_train, 16)

    hypervd_train = np.load("np_predictions/train/off_hypervd.npy")
    hypervd_train = np.repeat(hypervd_train, 16)

    macilsd_train = np.load("np_predictions/train/off_macil_sd.npy")
    macilsd_train = np.repeat(macilsd_train, 16)

    msbt_train = np.load("np_predictions/train/off_MSBT_train.npy")

    gt_train = np.load("list/newsplit/train/gt.npy")

    ## Test predictions
    hlnet_test = np.load("np_predictions/test/off_hl-net.npy")
    hlnet_test = np.repeat(hlnet_test, 16)

    cma_test = np.load("np_predictions/test/off_cma.npy")
    cma_test = np.repeat(cma_test, 16)

    hypervd_test = np.load("np_predictions/test/off_hypervd.npy")
    hypervd_test = np.repeat(hypervd_test, 16)

    macilsd_test = np.load("np_predictions/test/off_macil_sd.npy")
    macilsd_test = np.repeat(macilsd_test, 16)

    msbt_test = np.load("np_predictions/test/off_MSBT_test.npy")

    gt_test = np.load("list/newsplit/test/gt.npy")

    # Create a DataFrame by combining arrays as columns
    train_set = [hlnet_train, cma_train, hypervd_train, macilsd_train, msbt_train, gt_train]
    train_df = pd.DataFrame(train_set).T

    # Separate features and target variable
    features = train_df.iloc[:, :-1]  # All columns except the last
    target = train_df.iloc[:, -1]  # Last column

    # Convert features and target to tensors
    features_tensor = torch.from_numpy(features.values).float()
    target_tensor = torch.from_numpy(target.values).float()

    training = True
    if training:
        # Train the model
        # Train the meta-model on the combined feature matrix and the target values
        meta_model = LinearRegression()
        meta_model.fit(features, target)

    train_pred = meta_model.predict(features)
    np.save("np_predictions/train/metamodel.npy", train_pred)

    train_precision, train_recall, train_th = precision_recall_curve(list(gt_train), train_pred)
    train_pr_auc = auc(train_recall, train_precision)
    print('Train: offline pr_auc:{0:.4}; \n'.format(train_pr_auc))

    test_set = [hlnet_test, cma_test, hypervd_test, macilsd_test, msbt_test, gt_test]
    test_df = pd.DataFrame(test_set).T

    # Separate features and target variable
    test_features = test_df.iloc[:, :-1]  # All columns except the last
    test_targets = test_df.iloc[:, -1]  # Last column

    test_pred = meta_model.predict(test_features)
    test_precision, test_recall, test_th = precision_recall_curve(list(gt_test), test_pred)
    test_pr_auc = auc(test_recall, test_precision)
    print('Test: offline pr_auc:{0:.4}; \n'.format(test_pr_auc))
    np.save("np_predictions/test/metamodel.npy", test_pred)

    # features_tensor = torch.from_numpy(test_features.values).float()
    # # test_pred = model(features_tensor)
    # metamodel = MetaModel()
    # metamodel_pred = metamodel.predict(test_features.values)
    #
    # # precision, recall, th = precision_recall_curve(list(gt_test), test_pred.detach())
    # precision, recall, th = precision_recall_curve(list(gt_test), metamodel_pred)
    # pr_auc = auc(recall, precision)
    # print('offline pr_auc:{0:.4}; \n'.format(pr_auc))
