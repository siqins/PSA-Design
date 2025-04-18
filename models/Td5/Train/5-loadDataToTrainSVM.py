import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pickle
import time


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main(data_path, save_mode, model_path, file_path, log_path):
    # if save_mode:
    #     sys.stdout = Logger(log_path)
    log_list = []
    data_all = pd.read_csv(data_path).values
    data_train, data_test = train_test_split(data_all, test_size=0.10)
    print(data_train.shape, data_test.shape)
    log_list.append(str(data_train.shape) + '\t' + str(data_test.shape) + '\n')
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1].reshape(-1, 1)
    X_test = data_test[:, :-1]
    y_test = data_test[:, -1].reshape(-1, 1)

    model = SVR(kernel='rbf', C=200, gamma=0.03, epsilon=0.1)


    # Perform CV5 cross validation
    metrics = ['r2', 'neg_mean_absolute_error']
    results = cross_validate(model, X_train, y_train, cv=5, scoring=metrics, return_estimator=True,
                             return_train_score=True)
    # print(sorted(results.keys()))
    train_r2 = results['train_r2']
    validation_r2 = results['test_r2']
    train_neg_mae = results['train_neg_mean_absolute_error']
    validation_neg_mae = results['test_neg_mean_absolute_error']
    print('train R-square {}\nvalidation R-square {}'.format(train_r2, validation_r2))
    print('train Neg-MAE {}\nvalidation Neg-MAE {}'.format(train_neg_mae, validation_neg_mae))
    print('5-fold CV R-square: train_mean {}, train_std {}; validation_mean {}, validation_std {}'.format(
        np.mean(train_r2), np.std(train_r2), np.mean(validation_r2), np.std(validation_r2)))
    print('5-fold CV Neg-MAE: train_mean {}, train_std {};  validation_mean {}, validation_std {}'.format(
        np.mean(train_neg_mae), np.std(train_neg_mae), np.mean(validation_neg_mae), np.std(validation_neg_mae)))
    log_list.append('train R-square {}\nvalidation R-square {}\n'.format(train_r2, validation_r2))
    log_list.append('train Neg-MAE {}\nvalidation Neg-MAE {}\n'.format(train_neg_mae, validation_neg_mae))
    log_list.append(
        '5-fold CV R-square: train_mean {}, train_std {}; validation_mean {}, validation_std {}\n'.format(
            np.mean(train_r2), np.std(train_r2), np.mean(validation_r2), np.std(validation_r2)))
    log_list.append(
        '5-fold CV Neg-MAE: train_mean {}, train_std {};  validation_mean {}, validation_std {}\n'.format(
            np.mean(train_neg_mae), np.std(train_neg_mae), np.mean(validation_neg_mae), np.std(validation_neg_mae)))

    # Locate the model with the best performance (validation R2 maximum), take the model kernel as the prior distribution, retrain train all, and calculate the accuracy on test
    best_model = results['estimator'][int(np.where(validation_r2 == np.max(validation_r2))[0])]
    print(best_model)
    log_list.append(str(best_model) + '\n')
    best_model.fit(X_train, y_train)
    print(best_model)
    log_list.append(str(best_model) + '\n')
    y_pred_train = best_model.predict(X_train).reshape(-1, 1)
    y_pred_test = best_model.predict(X_test).reshape(-1, 1)
    print(r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test))
    print(mean_absolute_error(y_train, y_pred_train), mean_absolute_error(y_test, y_pred_test))
    log_list.append(str(r2_score(y_train, y_pred_train)) + '\t' + str(r2_score(y_test, y_pred_test)) + '\n')
    log_list.append(str(mean_absolute_error(y_train, y_pred_train)) + '\t' + str(
        mean_absolute_error(y_test, y_pred_test)) + '\n')
    Train_R2_list.append(str(r2_score(y_train, y_pred_train)))
    Train_MAE_list.append(str(mean_absolute_error(y_train, y_pred_train)))
    Test_R2_list.append(str(r2_score(y_test, y_pred_test)))
    Test_MAE_list.append(str(mean_absolute_error(y_test, y_pred_test)))

    if save_mode:
        # Save the trained model
        with open(model_path, 'wb') as fw:
            pickle.dump(best_model, fw)

        # Save training and test sets
        data_shuffle = np.vstack((data_train, data_test))
        shuffle_pd = pd.DataFrame(data_shuffle)
        shuffle_pd.to_csv(file_path, index=False)

        with open(log_path, 'w') as fw:
            for log in log_list:
                fw.write(log)


if __name__ == '__main__':
    xyDataPath = r'Test_X_Y.csv'
    modelSaveMode = True

    min_ = 0
    max_ = 1000

    Name = "SVM"

    Train_R2_list = []
    Test_R2_list = []
    Train_MAE_list = []
    Test_MAE_list = []
    for i in range(min_, max_):
        modelPath = r'./Model_SVM0.3/' + '_' + str(i)
        filePath = r'./Model_SVM0.3/' + '_shuffle_data_' + str(i) + '.csv'
        logPath = r'./Model_SVM0.3/' + '_Run_' + str(i) + '.txt'
        main(xyDataPath, modelSaveMode, modelPath, filePath, logPath)
        time.sleep(0.1)
    out = {'Train_RSquare': Train_R2_list,
            'Train_MAE': Train_MAE_list,
            'Test_RSquare': Test_R2_list,
            'Test_MAE': Test_MAE_list}
    out_pd = pd.DataFrame(out)
    outName = Name + '_Di_' + str(min_) + '-' + str(max_) + '200_0.03' '.csv'
    out_pd.to_csv(outName, index=False)

