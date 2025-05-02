from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: pd.Series = None
) -> tuple:
    """
    Делит признаки X и целевую переменную y на обучающую и тестовую выборки.

    Аргументы:
    - X: DataFrame признаков
    - y: Series целевой переменной
    - test_size: доля тестовой выборки
    - random_state: для воспроизводимости
    - stratify: Series для стратифицированного разбиения (обычно y)

    Возвращает:
    (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )