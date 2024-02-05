from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def logisticRegression(train, test, plot_show):
    train_encoded = pd.get_dummies(train, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])

    classes = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    display_classes = ['I', 'N', 'OV I', 'OV II', 'OB I', 'OB II', 'OB III']

    Y = train_encoded['NObeyesdad']
    X = train_encoded.drop(['NObeyesdad'], axis=1)

    # scale data with transformer that is robust to outliers
    transformer = RobustScaler().fit(X)
    X = transformer.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy}")

    if plot_show:
        cm = confusion_matrix(y_val, val_predictions, labels=classes)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='3.0f', ax = ax, cmap="viridis")

        ax.set_xlabel('Predicted labels') 
        ax.set_ylabel('True labels')     
        ax.xaxis.set_ticklabels(display_classes)   
        ax.yaxis.set_ticklabels(display_classes)
        ax.tick_params(axis='x', labelrotation=30)
        ax.tick_params(axis='y', labelrotation=0)
        plt.show()

    id = test['id']
    X_test = pd.get_dummies(test, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])
    X_test = transformer.transform(X_test)

    test_predictions = model.predict(X_test)
    DF = pd.DataFrame({'id': id , "NObeyesdad": test_predictions})
    DF.to_csv("submission.csv", index=False)