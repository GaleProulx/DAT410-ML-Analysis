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

    df = pd.DataFrame({'Male': data['% Males'].sum(), 'Female': data['% Females'].sum()}, index=['Genders'])
    ax = df.plot.bar(rot=0)
    plt.show()


if __name__ == "__main__":
    main()
    plot()
