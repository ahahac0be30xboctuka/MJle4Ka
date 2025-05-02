


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def prepare_scaled_data(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard"):
    """
    Масштабирование признаков:
    - method="standard": стандартизация через StandardScaler (среднее=0, std=1)
    - method="minmax": нормализация через MinMaxScaler (диапазон [0,1])
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Метод должен быть 'standard' или 'minmax'")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled