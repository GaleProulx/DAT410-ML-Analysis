# Author: Gale Proulx
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
# ALL CODE IN THIS FILE WAS CREATED BY GALE

# IMPORT DEPENDENCIES & SET CONFIGURATION
# ############################################################################
from pandas import ExcelWriter
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 100)


# FUNCTIONS
# ############################################################################
def correlation_matrix(df: pd.DataFrame):
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
    plt.show()


def cross_tab_kmeans(predictions: np.array, answers: list):
    cross_df = pd.DataFrame({'labels': predictions, 'answers': answers})
    crosstab = pd.crosstab(cross_df['labels'], cross_df['answers'])
    print('=== Cross Tabulation Results ===')
    print(crosstab)
    crosstab.to_csv('cross_tabulation_results.csv')


def decorrelate_data(df: pd.DataFrame, num_components=None) \
        -> pd.DataFrame:
    pca = PCA(n_components=num_components)
    pca.fit(df)

    if num_components is not None:
        columns = list(df.columns)[:num_components]
    else:
        columns = df.columns

    decorrelated_df = pd.DataFrame(pca.transform(df), columns=columns)
    print('=== Decorrelating Data ===')
    print('= Mean: ' + str(decorrelated_df.mean(axis=0).mean().round(2)))
    print('= Variance: ' + str(decorrelated_df.var(axis=0).mean().round(2)))
    print('==========================')

    return decorrelated_df


def dummy_transformation(df: pd.DataFrame, dummy_cols: list,
                         left_suffix='', right_suffix='') -> pd.DataFrame:
    for col in dummy_cols:
        dummy_df = pd.get_dummies(df[col])
        df.drop(columns=[col], inplace=True)
        df = df.join(dummy_df, lsuffix=left_suffix, rsuffix=right_suffix)

    return df


def export_results(df: pd.DataFrame, divider: str):
    if not os.path.exists('data'):
        os.mkdir('data')

    stats = ExcelWriter('data/Statistics_Report.xlsx')
    variances = ExcelWriter('data/Variances_Report.xlsx')
    cluster_dfs = ExcelWriter('data/Clusters_Report.xlsx')

    for value in df[divider].unique():
        value_df = df.loc[df[divider] == value]
        var_df = pd.DataFrame(data=value_df.var())
        value_df.describe().to_excel(stats, str(value) + '_stats')
        value_df.to_excel(cluster_dfs, str(value) + '_dataframe')
        var_df.to_excel(variances, str(value) + '_variance.csv')

    stats.save()
    variances.save()
    cluster_dfs.save()

    stats.close()
    variances.close()
    cluster_dfs.close()


def intrinsic_dimension(df: pd.DataFrame, num_components=None):
    pca = PCA(n_components=num_components)
    pca.fit_transform(df)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xticks(features)
    plt.xlabel('PCA Features')
    plt.ylabel('Variance')
    plt.title('Intrinsic Dimension Analysis')
    plt.show()


def kmeans_clusters(df: pd.DataFrame, clusters=3, seed=None) -> np.array:
    kmean = KMeans(n_clusters=clusters, random_state=seed)
    kmean.fit(df)

    return kmean.labels_


def kmeans_inertia(df: pd.DataFrame, max_clusters=10,
                   min_clusters=1, seed=None):
    clusters_range = range(min_clusters, max_clusters)
    inertias = list()

    for k in clusters_range:
        kmean = KMeans(n_clusters=k, random_state=seed)
        kmean.fit(df)
        inertias.append(kmean.inertia_)

    plt.plot(clusters_range, inertias, '-o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(clusters_range)
    plt.title('KMeans Inertia Analysis')
    plt.show()


def load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)

    return df


def load_iris_dataset(include_labels=True) -> pd.DataFrame:
    iris = datasets.load_iris()
    X = iris.data
    X_features = iris.feature_names

    df = pd.DataFrame(data=X, columns=X_features)
    if include_labels:
        y = iris.target
        y_target = iris.target_names
        df['labels'] = y
        df['named_labels'] = df['labels'].map({0: y_target[0],
                                              1: y_target[1],
                                              2: y_target[2]})

    return df


def log_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(np.log)


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(df)

    standardized_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    print('=== Standardizing Data ===')
    print('= Mean: ' + str(standardized_df.mean(axis=0).mean().round(2)))
    print('= Standard Deviation: ' + str(standardized_df
                                         .std(axis=0).mean().round(2)))
    print('==========================')

    return standardized_df


def string_to_unique_list(column: pd.Series) -> pd.DataFrame:
    unique_grade = list()
    for item in column.iteritems():
        grade_str = str(item[1])
        grades = grade_str.split(',')
        for grade in grades:
            if (grade not in unique_grade) and (grade != 'nan'):
                unique_grade.append(grade)

    unique_grade.sort()

    return unique_grade


def string_to_dummy_dataframe(df: pd.DataFrame, column: str,
                              uni_list: list) -> pd.DataFrame:
    zeroed_data = np.zeros((df.shape[1], len(uni_list)))
    dum_grade = pd.DataFrame(zeroed_data, columns=uni_list)

    for index, row in df.iterrows():
        school_grades = str(row[column]).split(',')
        for grade in school_grades:
            if grade != 'nan':
                dum_grade.loc[index, grade] = 1

    return dum_grade


def viz_histogram(df: pd.DataFrame, n_bins=30, separate_graphs=True,
                  title='', transparency=0.5):
    if separate_graphs:
        for col in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=n_bins)
            ax.set_title('{header}'.format(header=str(title + ': ')
                         if len(title) > 0 else '') + str(col))

    else:
        fig, ax = plt.subplots()
        for col in df.columns:
            ax.hist(df[col], label=str(col), bins=n_bins, alpha=transparency)
            ax.set_title(title)
            ax.legend()

    plt.show()


def viz_scatterplot(x: np.array, y: np.array, color=None, x_title='',
                    y_title='', title=''):
    if color is None:
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
    else:
        plt.scatter(x, y, c=color)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.legend()

    plt.show()


def viz_tsne(samples: pd.DataFrame, title='2D Visualization of Data',
             color=None, legend=False):
    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(samples)
    x = transformed[:, 0]
    y = transformed[:, 1]
    if legend:
        fig, ax = plt.subplots()
        ax.set_title(title)
        scatter = ax.scatter(x, y, alpha=0.5, c=color)
        ax.legend(*scatter.legend_elements(), title='Clusters')
        plt.show()
    else:
        plt.title(title)
        plt.scatter(x, y, c=color)
        plt.show()


# MAIN
# ############################################################################
def main() -> None:
    # PART I: Data Preparation
    # ########################################################################

    # [INITIAL STEP] Load data and only keep columns we want to analyze.
    df = load_dataset('MA_Public_Schools_2017.csv')
    keep_cols = [
        'PK_Enrollment', 'K_Enrollment',
        '1_Enrollment', '2_Enrollment', '3_Enrollment',
        '4_Enrollment', '5_Enrollment', '6_Enrollment',
        '7_Enrollment', '8_Enrollment', '9_Enrollment',
        '10_Enrollment', '11_Enrollment', '12_Enrollment',
        'SP_Enrollment', 'TOTAL_Enrollment',
        '% First Language Not English',
        '% English Language Learner',
        '% Students With Disabilities',
        '% High Needs', '% Economically Disadvantaged', '% African American',
        '% Asian', '% Hispanic', '% White', '% Native American',
        '% Native Hawaiian, Pacific Islander', '% Multi-Race, Non-Hispanic',
        '% Males', '% Females', 'Total # of Classes',
        'Average Class Size', 'Number of Students', 'Salary Totals',
        'Average Salary', 'FTE Count', 'In-District Expenditures',
        'Total In-district FTEs',
        'Average In-District Expenditures per Pupil',
        'Total Expenditures', 'Total Pupil FTEs',
        'Average Expenditures per Pupil',
        'Accountability and Assistance Level',
        'School Accountability Percentile (1-99)',
        'Progress and Performance Index (PPI) - All Students',
        'Progress and Performance Index (PPI) - High Needs Students',
        'District_Accountability and Assistance Level'
    ]
    df = df[keep_cols]

    # [TRANSFORMATIVE STEP] Create dummy columns for columns that need it.
    dummy_cols = ['Accountability and Assistance Level',
                  'District_Accountability and Assistance Level']
    df = dummy_transformation(df, dummy_cols, left_suffix='_School',
                              right_suffix='_District')
    
    # [INFORMATIVE STEP] Evaluate Nan values and remove empty columns.
    too_empty = ['# in Cohort',
        '% Graduated', '% Still in School', '% Non-Grad Completers',
        '% GED', '% Dropped Out', '% Permanently Excluded',
        'High School Graduates (#)', 'Attending Coll./Univ. (#)',
        '% Attending College', '% Private Two-Year',
        '% Private Four-Year', '% Public Two-Year',
        '% Public Four-Year', '% MA Community College',
        '% MA State University', '% UMass'
    ]
    
    keep_cols = [col for col in keep_cols if col not in too_empty]

    # [TRANSFORMATIVE STEP] Fill empty values with mean.
    df_cols = list(df.columns)
    for col in df_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    df.to_csv('cleaned_data.csv', index=False)
    
    
    # print(df.head())
    # print(df.columns)
    # print(df.shape)


if __name__ == "__main__":
    main()
