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

# IMPORT DEPENDENCIES & SET CONFIGURATION
# ############################################################################
from sklearn.model_selection import train_test_split

import pandas as pd

# FUNCTIONS
# ############################################################################
def import_train_test(filename: str, feature: str, train_size=0.33):
    df = pd.read_csv(filename)
    target = df[feature]
    features = df.drop([feature, 'Average In-District Expenditures per Pupil']
                       , axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                        test_size=train_size)
    
    return X_train, X_test, y_train, y_test
    
# MAIN
# ############################################################################
def main() -> None:
    X_train, X_test, y_train, y_test = \
        import_train_test('standardized_data.csv', 
                          'Average Expenditures per Pupil', train_size=0.50)
    
    print("Finn's Code Here...")

if __name__ == "__main__":
    main()