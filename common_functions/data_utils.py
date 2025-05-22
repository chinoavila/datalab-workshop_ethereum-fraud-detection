import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_and_preprocess_data(dataset_path,
                            target_column='FLAG',
                            test_size=0.3,
                            random_state=42):
    # Cargar dataset
    df = pd.read_csv(dataset_path)
    # Separar features y target
    X = pd.get_dummies(df.drop(target_column, axis=1))
    y = df[target_column]
    # Unir para limpiar nulos de manera consistente
    xy = pd.concat([X, y], axis=1)
    xy = xy.dropna()
    # Separar después de la limpieza
    X = xy.drop(target_column, axis=1)
    y = xy[target_column]
    # Dividir en train y test
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def normalize_features(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized, scaler
