import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier, RandomForestClassifier


def models_factory():
    return [runKnn, runRandomForest, runLDA, run_adaboost, runTrees, runMLPC,
            runLogistic, runSVM]


def runKnn(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by K-neares-neighbors and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("KNN models")
    scores = []
    for i in (list(range(4, 50, 5)) + [2, 3]):
        kn = KNeighborsClassifier(n_neighbors=i)
        scores.append(
            (kn, run_and_score(kn, X_train, Y_train, X_test, Y_test)))
    return scores


def runTrees(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by random forest and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("Trees models")
    scores = []
    for i in range(1, 100):
        dt = DecisionTreeClassifier(max_depth=i)
        scores.append(
            (dt, run_and_score(dt, X_train, Y_train, X_test, Y_test)))
    return scores


def runSVM(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by SVM and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("SVM models")
    types = ['linear', 'poly']
    multi_ways = ['ovo', 'ovr']
    scores = []
    for i in range(1, 4):
        for tpe in types:
            for mul_func in multi_ways:
                svm = SVC(gamma=i, kernel=tpe,
                          decision_function_shape=mul_func, probability=True)
                scores.append((svm,
                               run_and_score(svm, X_train, Y_train, X_test,
                                             Y_test)))
    return scores


def runLDA(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by LDA and checks it by the test set.8

    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: list of tuples of (models, score).
    """
    print("LDA model")
    scores = []
    set_lda = ['svd', 'lsqr']
    for ld in set_lda:
        lda = LinearDiscriminantAnalysis(solver=ld)
        scores.append(
            (lda, run_and_score(lda, X_train, Y_train, X_test, Y_test)))
    return scores


def perf_measure(y_actual, y_hat):
    """
    this function gets two vectors of predicted y and actual y ant return the
    measurements: (TP, FP, TN, FN)
    :param y_actual:
    :param y_hat:
    :return: (TP, FP, TN, FN)
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == -1:
            TN += 1
        if y_hat[i] == -1 and y_actual[i] != y_hat[i]:
            FN += 1
    return (TP, FP, TN, FN)


def run_and_score(model, X_train, y_train, X_test, y_test):
    """
    this function run the model on X_train, y_train and score it according to
    X_test, y_test.
    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: dictionary with the measurement: PRECISION,TPR,TNR
    """
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    TP, FP, TN, FN = perf_measure(y_test, y_predict)
    precision = (TP + TN) / (TP + FP + TN + FN)
    TNR = TN / (FP + TN)
    TPR = TP / (FN + TP)
    return {"PRECISION": precision, "TPR": TPR, "TNR": TNR}


def runMLPC(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by MLCP and checks it by the test set.

    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: list of tuples of (models, score).
    """
    print("MLPC model")
    scores = []
    alphas = [1e-4, 1e-5, 1e-6, 1e-7]
    layers = [(5, 2), (6, 2), (7, 2), (5, 3), (6, 3), (7, 3), (8, 2), (9, 2)]
    for alpha in alphas:
        for layer in layers:
            model = MLPClassifier(random_state=1, alpha=alpha,
                                  hidden_layer_sizes=layer, max_iter=600)
            scores.append((model,
                           run_and_score(model, X_train, Y_train, X_test,
                                         Y_test)))
    return scores


def runRandomForest(X_train, y_train, X_test, y_test):
    """
    trains the train set by Random Forest and checks it by the test set.

    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: list of tuples of (models, score).
    """
    scores = []
    print("Random forest model")
    for nEstim in range(5, 101, 5):
        for i in range(1, 11):
            model = RandomForestClassifier(n_estimators=nEstim, max_depth=i)
            scores.append(
                (
                model, run_and_score(model, X_train, y_train, X_test, y_test)))
    return scores


def runBagging(models, X_train, y_train, X_test, y_test):
    """
    splits to k folds an the returns the best model and its average.
    :param k: number of folds.
    :param X: data set, numpy array.
    :param y: vector response, numpy array.
    :param models: different models for prediction
    :return: list of tuples of (models, score).
    """
    scores = []
    print("Bagging")
    for model in models:
        bg = BaggingClassifier(model[0], max_samples=0.7, max_features=1.0,
                               n_estimators=20)
        scores.append(
            (bg, run_and_score(bg, X_train, y_train, X_test, y_test)))
    return scores


def run_adaboost(X_train, Y_train, X_test, Y_test):
    """
    runs adaboost on the data.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: list of tuples of (models, score).
    """
    print("AdaBoost")
    scores = []
    for i in range(30, 101, 10):
        model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=i)
        scores.append(
            (model, run_and_score(model, X_train, Y_train, X_test, Y_test)))
    return scores


def runLogistic(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by Logistic regression classifier and checks it by the test set.

    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: list of tuples of (models, score).
    """
    print("Logistic model")
    scores = []
    saga_lst = ["elasticnet", 'l2', "l1", "none"]
    for regTerm in saga_lst:
        for c in range(1, 4):
            if regTerm == "elasticnet":
                for i in range(1, 10):
                    log = LogisticRegression(solver='saga', penalty=regTerm,
                                             C=c, max_iter=1000,
                                             l1_ratio=0.1 * i)
                    scores.append((log,
                                   run_and_score(log, X_train, Y_train, X_test,
                                                 Y_test)))
            else:
                log = LogisticRegression(solver='saga', penalty=regTerm, C=c,
                                         max_iter=1000)
                scores.append((log,
                               run_and_score(log, X_train, Y_train, X_test,
                                             Y_test)))
    return scores


def find_best_model(scores, per=False):
    """
    this function finds the best model according to the score tuple list
    :param scores:
    :param per: a flag that determine if we evaluate the score according to TPR
    or PRECISION best performence model
    :return: the best model according to the per flag
    """
    key = (lambda tup: tup[-1]["TPR"]) if not per else (
        lambda tup: tup[-1]["PRECISION"])
    return max(scores, key=key)


def run_models(X_train, y_train, X_test, y_test, flag_score=False):
    """
    this function runs the
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param flag_score:
    :return:
    """
    lst_models = []
    for run_model in models_factory():
        lst_models.append(
            find_best_model(run_model(X_train, y_train, X_test, y_test),flag_score))
    lst_models += runBagging(lst_models, X_train, y_train, X_test, y_test)
    lst_models += run_voting(lst_models,X_train, y_train, X_test, y_test)
    return lst_models


def run_voting(models, X_train, Y_train, X_test, Y_test):
    models_to_voting = [(str(i), models[i][0]) for i in
                        range(len(models))]
    scores = []
    mod = VotingClassifier(estimators=models_to_voting, voting="hard")
    scores.append((mod, run_and_score(mod, X_train, Y_train, X_test, Y_test)))
    mod = VotingClassifier(estimators=models_to_voting[1:], voting="soft")
    scores.append((mod, run_and_score(mod, X_train, Y_train, X_test, Y_test)))
    ###########################################################################
    # for i in [1.2, 1.4, 1.6, 1.8, 2]:
    #     mod = VotingClassifier(estimators=sort_by_fnr, voting="soft",
    #                            weights=[j * i for j in
    #                                     range(1, len(models) + 1)])
    #     mod2 = VotingClassifier(estimators=sort_by_per, voting="soft",
    #                             weights=[j * i for j in
    #                                      range(1, len(models) + 1)])
    #     scores.append(
    #         (mod, run_and_score(mod, X_train, Y_train, X_test, Y_test)))
    #     scores.append(
    #         (mod2, run_and_score(mod2, X_train, Y_train, X_test, Y_test)))
    ###########################################################################
    return scores
