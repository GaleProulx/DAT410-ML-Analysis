# Author: Gale Proulx and Finn Jensen
# Class:  DAT-310-01
# Certification of Authenticity:
# I certify that this is my work and the DAT-330 class work,
# except where I have given fully documented references to the work
# of others. I understand the definition and consequences of plagiarism
# and acknowledge that the assessor of this assignment may, for the purpose
# of assessing this assignment reproduce this assignment and provide a
# copy to another member of academic staff and / or communicate a copy of
# this assignment to a plagiarism checking service(which may then retain a
# copy of this assignment on its database for the purpose of future
# plagiarism checking).
#

# IMPORT DEPENDENCIES & SET CONFIGURATION
# ############################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import accuracy


# FUNCTIONS
# ############################################################################
# Gale
def import_train_test(filename: str, feature: str, train_size=0.33):
    df = pd.read_csv(filename)
    target = df[feature]
    features = df.drop([feature, 'Average In-District Expenditures per Pupil'],
                       axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=train_size)

    return X_train, X_test, y_train, y_test


# Finn
def build_model(nn=16, nl=5, loss="mean_squared_error",
                optimizer="rmsprop", activation="relu"):
    model = Sequential()
    model.add(Dense(units=55, input_dim=55, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="Adam", loss="mean_squared_error",
                  metrics=['mean_squared_error'])
    return model


# Finn
def hyperparameter_tuning(X_train, y_train):
    model = KerasRegressor(build_fn=build_model, verbose=2)
    params = {'activation': ['relu', 'tanh', 'elu', 'softmax',
                             'selu'], 'batch_size': [24, 32, 128, 256],
              'epochs': [25, 50, 100, 200], 'nl': [1, 2, 5, 9],
              'nn': [20, 60, 128, 256], 'loss': ['mean_squared_error',
                                                 'huber_loss',
                                                 'binary_crossentropy',
                                                 'categorical_crossentropy'],
              'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']}

    random_search = RandomizedSearchCV(model, param_distributions=params,
                                       cv=3, n_jobs=-1, verbose=1)
    fitted_model = random_search.fit(X_train, y_train, verbose=2)
    return fitted_model


# MAIN
# ############################################################################
# Gale and Finn
def main() -> None:
    X_train, X_test, y_train, y_test = import_train_test('cleaned_data.csv',
                                                         'Average Expenditures'
                                                         'per Pupil',
                                                         train_size=0.50)

    model = build_model()
    model.fit(X_train, y_train, verbose=1, epochs=1500)

    y_pred = model.predict(X_test)

    print("Model Info")
    print("-------------------------------------------------")
    print('Accuracy: {}'.format(model.evaluate(X_test, y_test)))
    # print("Tuned Model Parameters: {}".format(model.best_params_))
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
