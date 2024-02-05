import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def preprocess_data(train, test):
    # By running this code we can see that there are no null values in the dataset
    print(train.isnull().sum())
    print(test.isnull().sum())
    
    # We can see that the dataset has a mix of continuous and categorical features.
    for column in test.columns:
        print(column, train[column].nunique(), test[column].nunique())

    # One categorical feature (CALC) has different number of unique values between
    # train and test sets, so we need to investigate.
    print(train['CALC'].value_counts())
    print(test['CALC'].value_counts())

    # We can see that the test column has 'Always' value that the train column 
    # doesn't.

    # plot correlation table of continuous features:
    correlation_table = train[train.select_dtypes(include=['int64', 'float64']).columns].corr()
    sns.heatmap(correlation_table, annot=True, fmt='.3f', cmap="viridis")
    plt.show()
    plt.clf()

    # some correlation, but not much to worry about

    classes = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    display_classes = ['I', 'N', 'OV I', 'OV II', 'OB I', 'OB II', 'OB III']

    train.NObeyesdad=pd.Categorical(train.NObeyesdad,categories=classes)
    train=train.sort_values('NObeyesdad')

    continuous = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    for column in continuous:
        _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        plt.setp(axes, xticklabels=display_classes)
        sns.stripplot(ax = axes[0], data=train, x='NObeyesdad', y=column)
        sns.boxplot(ax = axes[1],data=train, x='NObeyesdad', y=column, orient='x')
        plt.show()

    for column in categorical:
        sns.countplot(data=train, x='NObeyesdad', hue=column)
        plt.show()