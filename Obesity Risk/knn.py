from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
import pandas as pd


def knn(train, test, plot_show):
    train_encoded = pd.get_dummies(train, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])

    classes = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    display_classes = ['I', 'N', 'OV I', 'OV II', 'OB I', 'OB II', 'OB III']

    Y = train_encoded['NObeyesdad']
    X = train_encoded.drop(['NObeyesdad'], axis=1)

    # scale data with transformer that is robust to outliers
    transformer = RobustScaler().fit(X)
    X = transformer.transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors = 50)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy}")
