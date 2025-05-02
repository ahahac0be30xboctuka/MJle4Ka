from typing import Any, Tuple, Optional
import numpy as np

def predict(model: Any, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Делает предсказания для переданных данных X.

    Args:
        model: обученная модель с методами predict и, возможно, predict_proba.
        X: данные для предсказаний (numpy array, pandas DataFrame или аналогичные).

    Returns:
        Tuple:
        - y_pred: массив предсказанных меток (np.ndarray).
        - y_proba: массив вероятностей классов (np.ndarray) или None, если модель не поддерживает predict_proba.
    """
    # Преобразуем X в numpy array при необходимости
    try:
        X_arr = np.asarray(X)
    except Exception:
        X_arr = X

    y_pred = model.predict(X_arr)
    try:
        y_proba = model.predict_proba(X_arr)
    except AttributeError:
        y_proba = None

    return y_pred, y_proba