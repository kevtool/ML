import pandas as pd
from preprocess import preprocess_data
from randomForest import randomForest
from logisticReg import logisticRegression
from knn import knn
from xgb import xgboost

def main():
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')

    preprocess_data(train, test)

    # data preprocessing: the test data has two 'Always' labels in the CALC column
    # the 'Always' label is not found in the CALC column of the training data.
    test['CALC']=test['CALC'].replace('Always','Frequently')

    #randomForest(train, test, False)
    #logisticRegression(train, test, False)
    #knn(train, test, True)
    #xgboost(train, test)
    
if __name__ == "__main__":
	main()