# Author: Finn Jensen
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
# ALL CODE IN THIS FILE WAS CREATED BY FINN
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('MA_Public_Schools_2017.csv')

    for colName, colData in data.iteritems():
        if colData.isnull().values.any():
            print('Column: ' + colName + ' Contains Null')
        print('----------------------------')
        print(colName)
        print(colData.describe())
        print('----------------------------')


def plot():
    data = pd.read_csv('MA_Public_Schools_2017.csv')

    df = pd.DataFrame({'Male': data['% Males'].sum(),
                       'Female': data['% Females'].sum()}, index=['Genders'])
    ax = df.plot.bar(rot=0)
    plt.show()


if __name__ == "__main__":
    main()
    plot()
