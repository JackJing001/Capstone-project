# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:33:48 2021

"""
try:
    import ml_metrics
except ModuleNotFoundError:
    print("Please install ml_metrics library with : !pip install ml_metrics")
import numpy as np
import pandas as pd


def single_map(Y_train, train_predictions, top_k=5):
    # Function to calculate MAP@k score

    # Sort probabilities in ascending order
    train_ind = np.argsort(train_predictions, axis=1)

    # Extract only top_k predictions
    train_max_indices = train_ind[:, -top_k:]

    # Store indices of the actually cited papers
    train_actual_ind = np.nonzero(Y_train.values)

    # Create dataframes to store indices of cited papers
    actual_train = pd.DataFrame(
        index=train_actual_ind[0],
        data=train_actual_ind[1],
        columns=["indices"],
    )

    # Group by index to store citations for each paper in one row
    actual_train_indices = actual_train.groupby(actual_train.index)[
        "indices"
    ].apply(list)

    # Calculate AP@K for each row
    train_AP_scores = []

    for i in range(len(Y_train)):
        AP_temp = ml_metrics.apk(
            actual_train_indices.iloc[i], train_max_indices[i], top_k
        )
        train_AP_scores.append(AP_temp)

    # Calculate MAP@K
    train_MAP = np.array(train_AP_scores).mean()

    return train_MAP


def single_recall(Y_train, train_predictions, top_k=5):
    # Function to calculate MAP@k score

    # Sort probabilities in ascending order
    train_ind = np.argsort(train_predictions, axis=1)

    # Extract only top_k predictions
    train_max_indices = train_ind[:, -top_k:]

    # Store indices of the actually cited papers
    train_actual_ind = np.nonzero(Y_train.values)

    # Create dataframes to store indices of cited papers
    actual_train = pd.DataFrame(
        index=train_actual_ind[0],
        data=train_actual_ind[1],
        columns=["indices"],
    )

    # Group by index to store citations for each paper in one row
    actual_train_indices = actual_train.groupby(actual_train.index)[
        "indices"
    ].apply(list)

    # Calculate Recall@K for each row
    train_recall_scores = []

    for i in range(len(Y_train)):
        recall_temp = len(
            set(actual_train_indices.iloc[i]).intersection(
                set(train_max_indices[i])
            )
        ) / len(actual_train_indices.iloc[i])
        train_recall_scores.append(recall_temp)

    # Calculate Mean Recall@K
    train_recall = np.array(train_recall_scores).mean()
    return train_recall


def MAP_score(Y_train, Y_test, train_predictions, test_predictions, top_k=5):
    # Function to calculate MAP@k score

    # Sort probabilities in ascending order
    train_ind = np.argsort(train_predictions, axis=1)
    test_ind = np.argsort(test_predictions, axis=1)

    # Extract only top_k predictions
    train_max_indices = train_ind[:, -top_k:]
    test_max_indices = test_ind[:, -top_k:]

    # Store indices of the actually cited papers
    train_actual_ind = np.nonzero(Y_train.values)
    test_actual_ind = np.nonzero(Y_test.values)

    # Create dataframes to store indices of cited papers
    actual_train = pd.DataFrame(
        index=train_actual_ind[0],
        data=train_actual_ind[1],
        columns=["indices"],
    )
    actual_test = pd.DataFrame(
        index=test_actual_ind[0], data=test_actual_ind[1], columns=["indices"]
    )

    # Group by index to store citations for each paper in one row
    actual_train_indices = actual_train.groupby(actual_train.index)[
        "indices"
    ].apply(list)
    actual_test_indices = actual_test.groupby(actual_test.index)[
        "indices"
    ].apply(list)

    # Calculate AP@K for each row
    train_AP_scores = []
    test_AP_scores = []

    for i in range(len(Y_train)):
        AP_temp = ml_metrics.apk(
            actual_train_indices.iloc[i], train_max_indices[i], top_k
        )
        train_AP_scores.append(AP_temp)

    for i in range(len(Y_test)):
        AP_temp = ml_metrics.apk(
            actual_test_indices.iloc[i], test_max_indices[i], top_k
        )
        test_AP_scores.append(AP_temp)

    # Calculate MAP@K
    train_MAP = np.array(train_AP_scores).mean()
    test_MAP = np.array(test_AP_scores).mean()

    return train_MAP, test_MAP


def Recall_score(
    Y_train, Y_test, train_predictions, test_predictions, top_k=5
):
    # Function to calculate MAP@k score

    # Sort probabilities in ascending order
    train_ind = np.argsort(train_predictions, axis=1)
    test_ind = np.argsort(test_predictions, axis=1)

    # Extract only top_k predictions
    train_max_indices = train_ind[:, -top_k:]
    test_max_indices = test_ind[:, -top_k:]

    # Store indices of the actually cited papers
    train_actual_ind = np.nonzero(Y_train.values)
    test_actual_ind = np.nonzero(Y_test.values)

    # Create dataframes to store indices of cited papers
    actual_train = pd.DataFrame(
        index=train_actual_ind[0],
        data=train_actual_ind[1],
        columns=["indices"],
    )
    actual_test = pd.DataFrame(
        index=test_actual_ind[0], data=test_actual_ind[1], columns=["indices"]
    )

    # Group by index to store citations for each paper in one row
    actual_train_indices = actual_train.groupby(actual_train.index)[
        "indices"
    ].apply(list)
    actual_test_indices = actual_test.groupby(actual_test.index)[
        "indices"
    ].apply(list)

    # Calculate Recall@K for each row
    train_recall_scores = []
    test_recall_scores = []

    for i in range(len(Y_train)):
        recall_temp = len(
            set(actual_train_indices.iloc[i]).intersection(
                set(train_max_indices[i])
            )
        ) / len(actual_train_indices.iloc[i])
        train_recall_scores.append(recall_temp)

    for i in range(len(Y_test)):
        recall_temp = len(
            set(actual_test_indices.iloc[i]).intersection(
                set(test_max_indices[i])
            )
        ) / len(actual_test_indices.iloc[i])
        test_recall_scores.append(recall_temp)

    # Calculate Mean Recall@K
    train_recall = np.array(train_recall_scores).mean()
    test_recall = np.array(test_recall_scores).mean()

    return train_recall, test_recall
