import numpy as np
import pandas as pd
import Models
import pickle

DATA_PATH = "C:/Users/Owner/Desktop/CS/algoAsATool/predictor/cow_data.csv"
FILENAME_PICKLE_KET = "pickle_ket.sav"
FILENAME_PICKLE_MET = "pickle_met.sav"
NUM_OF_DAYS = 14
MEANS_PER_DAY = []
CORR_FEATURES_KET = {'SUM_GEN_RST_LST_avg_dif', 'SUM_GEN_RST_LST_max_dif',
                     'LACTATION_NUMBER', 'SUM_GEN_RST_LST_std_dif',
                     'SUM_GEN_RST_LST_max_val', 'SUM_GEN_RST_LST_mean',
                     'SUM_RUM_LST_min_val', 'SUM_RUM_LST_mean',
                     'SUM_RUM_LST_max_dif', 'SUM_RUM_LST_std',
                     'SUM_GEN_RST_LST_min_val', 'SUM_GEN_RST_LST_std',
                     'SUM_ACT_LST_mean', 'SUM_ACT_LST_min_val',
                     'SUM_ACT_LST_std', 'SUM_EAT_LST_min_val',
                     'SUM_EAT_LST_max_val', 'SUM_EAT_LST_mean',
                     'SUM_RUM_LST_avg_dif', 'SUM_RUM_LST_std_dif',
                     "SUM_RUM_LST_day_1", "SUM_RUM_LST_day_2",
                     "SUM_RUM_LST_day_3", "SUM_RUM_LST_day_4",
                     "SUM_RUM_LST_day_5", "SUM_RUM_LST_day_6",
                     "SUM_RUM_LST_day_7", "SUM_RUM_LST_day_8",
                     "SUM_RUM_LST_day_9", "SUM_RUM_LST_day_10",
                     "SUM_RUM_LST_day_11", "SUM_RUM_LST_day_12",
                     "SUM_RUM_LST_day_13",
                     "SUM_GEN_LST_day_1", "SUM_GEN_LST_day_2",
                     "SUM_GEN_LST_day_3", "SUM_GEN_LST_day_4",
                     "SUM_GEN_LST_day_5", "SUM_GEN_LST_day_6",
                     "SUM_GEN_LST_day_7", "SUM_GEN_LST_day_8",
                     "SUM_GEN_LST_day_9", "SUM_GEN_LST_day_10",
                     "SUM_GEN_LST_day_11", "SUM_GEN_LST_day_12",
                     "SUM_GEN_LST_day_13"}
CORR_FEATURES_MET = {"LACTATION_NUMBER", 'SUM_GEN_RST_LST_avg_dif',
                     'SUM_GEN_RST_LST_max_dif',  ################
                     'SUM_GEN_RST_LST_std_dif', 'SUM_ACT_LST_max_val',
                     'SUM_GEN_RST_LST_max_val', 'SUM_EAT_LST_avg_dif',
                     'SUM_GEN_RST_LST_mean', 'SUM_RUM_LST_min_val',
                     'SUM_EAT_LST_std', 'SUM_RUM_LST_mean',
                     'SUM_RUM_LST_max_dif', 'SUM_RUM_LST_std',
                     'SUM_GEN_RST_LST_min_val', 'SUM_GEN_RST_LST_std',
                     'SUM_ACT_LST_mean', 'SUM_ACT_LST_min_val',
                     'SUM_ACT_LST_std', 'SUM_EAT_LST_min_val',
                     'SUM_EAT_LST_max_dif', 'SUM_EAT_LST_mean',
                     'SUM_RUM_LST_avg_dif', 'SUM_EAT_LST_std_dif',
                     'SUM_RUM_LST_std_dif',
                     "SUM_RUM_LST_day_1", "SUM_RUM_LST_day_2",
                     "SUM_RUM_LST_day_3", "SUM_RUM_LST_day_4",
                     "SUM_RUM_LST_day_5", "SUM_RUM_LST_day_6",
                     "SUM_RUM_LST_day_7", "SUM_RUM_LST_day_0",
                     "SUM_GEN_LST_day_1", "SUM_GEN_LST_day_2",
                     "SUM_GEN_LST_day_3", "SUM_GEN_LST_day_4",
                     "SUM_GEN_LST_day_0"}
LST_FEATURES_NAME = ["SUM_RUM_LST", "SUM_EAT_LST", "SUM_ACT_LST",
                     "SUM_GEN_RST_LST"]
NOT_NORMALIZED_FEATURES = ["KET", "MET", "REGISTRATION_KEY",
                           "LACTATION_NUMBER"]


def make_lactation_dummies(data_frame):
    func = lambda x: x if x <= 2 else 3
    data_frame["LACTATION_NUMBER"] = data_frame["LACTATION_NUMBER"].apply(func)
    dumm = pd.get_dummies(data_frame['LACTATION_NUMBER'])
    del data_frame['LACTATION_NUMBER']
    return pd.concat([data_frame, dumm], axis=1)


def load_data_make_features(data_frame, ket=True):

    # the next line drop sam0ples with missing data
    data_frame.dropna(inplace=True)
    list_lambda = lambda lst: [int(i) for i in
                               (lst[1:-2].replace(' ', '')).split(
                                   ".")][41:55]
    for feature in LST_FEATURES_NAME:
        data_frame[feature] = data_frame[feature].map(list_lambda)

    # the next lines filtered the "LACTATION_NUMBER" "bedrooms" and "floors" features
    data_frame = data_frame[
        data_frame["LACTATION_NUMBER"].isin(list(range(1, 14)))]

    # the next lines filtered the lst_features with len != NUM_OF_DAYS

    len_lambda = (lambda x: len(x) != NUM_OF_DAYS)

    data_frame.drop(data_frame[
                        ((data_frame.SUM_RUM_LST.apply(len_lambda)) |
                         (data_frame.SUM_ACT_LST.apply(len_lambda)) |
                         (data_frame.SUM_EAT_LST.apply(len_lambda)) |
                         (data_frame.SUM_GEN_RST_LST.apply(
                             len_lambda)))].index, inplace=True)
    data_frame = add_features_according_lst(data_frame, ket)
    # correlation_finder(data_frame, "KET")
    # correlation_finder(data_frame, "MET")
    set_search = CORR_FEATURES_KET.union(
        {"KET"}) if ket else CORR_FEATURES_MET.union({"MET"})
    for col in data_frame.columns:
        if col in set_search: continue
        del data_frame[col]
    data_frame = normalize_features(data_frame)
    data_frame = make_lactation_dummies(
        data_frame)
    return data_frame


def calculate_means_per_day(data, feature):
    for i in range(NUM_OF_DAYS):
        lst_i = data[feature].apply(lambda feature: feature[i])
        mean = lst_i.mean()
        x = lst_i.apply(lambda x: (abs(x - mean) ** 2))  # todo: change
        data[feature + "_day_" + str(i)] = x


def add_features_according_lst(data, ket=True):
    """
    this function adds features according to the behaviors list : mean, std,
    max_vals and average and max difference between samples
    :param data: numpy X matrix
    :return: the X matrix after addition of features
    """
    lambda_dif = lambda lst: [abs(lst[i - 1] - lst[i]) for i in
                              range(1, len(lst))]
    lambda_max_dif = lambda lst: np.max(lambda_dif(lst))
    lambda_std_dif = lambda lst: np.std(lambda_dif(lst))
    lambda_avg_dif = lambda lst: np.mean(lambda_dif(lst))
    for lst in LST_FEATURES_NAME:
        set_search = CORR_FEATURES_KET if ket else CORR_FEATURES_MET
        if (lst + "_mean") in set_search:
            means = data[lst].apply(np.mean)
            data[lst + "_mean"] = means

        if (lst + "_std") in set_search:
            std = data[lst].apply(np.std)
            data[lst + "_std"] = std

        if (lst + "_max_val") in set_search:
            max_val = data[lst].apply(np.max)
            data[lst + "_max_val"] = max_val

        if (lst + "_min_val") in set_search:
            min_val = data[lst].apply(np.min)
            data[lst + "_min_val"] = min_val

        if (lst + "_max_dif") in set_search:
            max_dif = data[lst].apply(lambda_max_dif)
            data[lst + "_max_dif"] = max_dif

        if (lst + "_avg_dif") in set_search:
            avg_dif = data[lst].apply(lambda_avg_dif)
            data[lst + "_avg_dif"] = avg_dif

        if (lst + "_std_dif") in set_search:
            std_dif = data[lst].apply(lambda_std_dif)
            data[lst + "_std_dif"] = std_dif
        calculate_means_per_day(data, lst)
        del data[lst]
    return data


def normalize_features(data):
    """
    this function normalize the data to the range of [0,1]
    :param data: numpy X matrix
    :return: the X matrix after addition of features
    """
    result = data.copy()
    for feature_name in data.columns:
        if feature_name in (NOT_NORMALIZED_FEATURES):
            continue
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        x = (data[feature_name] - min_value) / (max_value - min_value)
        result[feature_name] = x
    return result


def pre_process_train(data_frame, ket=True):
    """
    this separates the data to X and to response vectors of met and ket
    :param data_frame: pandas df
    :return: numpy matrix X , y_ket, y_met
    """

    data = load_data_make_features(data_frame, ket)
    y = data["KET"] if ket else data["MET"]
    to_del = "KET" if ket else "MET"
    del data[to_del]
    # the next line switch the true vals to 1 and the false to -1
    lamb1 = lambda x: 1 if x else -1
    y = np.vectorize(lamb1)(y.to_numpy())
    return data.to_numpy(), y


def create_model(X_train, y_train, X_test, y_test, flag_score=False):
    """
    this function select the best model of classification base on the X_train
    and y_train data, and the score of the model on X_test with y_test label
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: the model with the best performance
    """
    lst_models = Models.run_models(X_train, y_train, X_test, y_test, flag_score)
    print(lst_models)
    best_model = Models.find_best_model(lst_models, flag_score)
    print(best_model)
    return best_model


def train_and_save(data_frame, pre=False):
    """
    this function gets dataframe, cleans it, add features, train model base on
    it and save the best model for classification on met and ket disease and
    save it to a pickle file
    :param data_frame: pandas df
    :return: None
    """
    for ket_flag in [True, False]:
        data_frame_copy = data_frame.copy()
        X, y = pre_process_train(data_frame_copy, ket_flag)
        # the next line is an array with the test indices for the very end of the process
        indices_test = list(range(0, len(X), 4))
        X_test, y_test = X[indices_test, :], y[indices_test]

        X_train, y_train = np.delete(X, indices_test, 0), \
                           np.delete(y, indices_test, 0)

        # the next line is an array with the validation indices for the train proccess
        indices_train_validation = list(range(0, len(X_train), 4))
        X_train_t, y_t_t = X_train[indices_train_validation, :], \
                           y_train[indices_train_validation]
        X_train_v, y_t_v = np.delete(X_train, indices_train_validation, 0), \
                           np.delete(y_train, indices_train_validation, 0)
        model = create_model(X_train_t, y_t_t, X_train_v, y_t_v, pre)
        file_name = FILENAME_PICKLE_KET if ket_flag else FILENAME_PICKLE_MET
        pickle.dump(model, open(file_name, 'wb'))


def correlation_finder(data_frame, label):
    """
    this function calculates the correlation between the data_frame features
    and the label and saves it to csv
    :param data_frame:
    :param label:
    :return: set of correlation that there abs val exceed 0.1
    """
    X = data_frame.copy()
    for col in data_frame.columns:
        X[col] = X[col].astype('category').cat.codes
    cols = X.columns[1:]
    corr = X[X.columns[1:]].corr()[label][:]
    lst_token = []
    final_set = set()
    for i in corr:
        flag = True if abs(i) >= 0.1 else False
        lst_token.append(flag)
    for i in range(len(cols)):
        if cols[i] not in {"KET", "MET"} and lst_token[i]:
            final_set.add(cols[i])
    print(final_set)
    corr.to_csv("correlation_" + label + ".csv")
    return final_set


if __name__ == '__main__':
    # the next line calculate the correlation

    ###########################################################################
    # data_frame_ket = pd.read_csv(DATA_PATH)
    # data_frame_met = pd.read_csv(DATA_PATH)
    # X_k = load_data_make_features(data_frame_ket)
    # X_m = load_data_make_features(data_frame_met, False)
    # set_corr_ket = correlation_finder(X_k, "KET")
    # set_corr_met = correlation_finder(X_m, "MET")
    # print(set_corr_ket)
    # print(set_corr_met)
    ###########################################################################
    data_frame = pd.read_csv(DATA_PATH)
    train_and_save(data_frame) #can save model that prefer accuracy on sensitivity
