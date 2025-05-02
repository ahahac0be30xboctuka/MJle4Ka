import os
from termcolor import colored
import time
import numpy as np
from typing import Dict, Any, Optional, Sequence
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from src.features.select_feature import prepare_scaled_data
import pandas as pd

def tune_model(
    model: Any,
    param_grid: Dict[str, Any],
    X_train: Sequence,
    y_train: Sequence,
    average: str = 'macro',
    cv: int = 5
) -> Dict[str, Any]:
    """
    Подбирает лучшие гиперпараметры для переданной модели.

    Args:
        model: объект модели sklearn.
        param_grid: сетка параметров для поиска.
        X_train: обучающие признаки.
        y_train: обучающие метки.
        average: тип усреднения для метрики f1 ('binary' или 'macro').
        cv: число фолдов для кросс-валидации.

    Returns:
        Словарь лучших параметров.
    """
    scoring = 'f1' if average == 'binary' else f"f1_{average}"
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"✅ {type(model).__name__}: лучшие параметры — {grid.best_params_}, f1 = {grid.best_score_:.4f}")
    return grid.best_params_

def get_models(
    best_logreg: Dict[str, Any],
    best_knn: Dict[str, Any],
    best_rf: Dict[str, Any],
    best_gb: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Создает и возвращает словарь моделей с переданными параметрами.

    Args:
        best_logreg: параметры для LogisticRegression.
        best_knn: параметры для KNeighborsClassifier.
        best_rf: параметры для RandomForestClassifier.
        best_gb: параметры для GradientBoostingClassifier.

    Returns:
        Словарь имя_модели -> инстанс модели.
    """
    return {
        'Logistic Regression': LogisticRegression(**best_logreg, max_iter=1000, random_state=42),
        'k-NN': KNeighborsClassifier(**best_knn),
        'Random Forest': RandomForestClassifier(**best_rf, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(**best_gb, random_state=42)
    }

def train_and_evaluate(
    models: Dict[str, Any],
    X_train_dict: Dict[str, Sequence],
    X_test_dict: Dict[str, Sequence],
    y_train: Sequence,
    y_test: Sequence,
    average: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Обучает переданные модели и оценивает их по метрикам.

    Args:
        models: словарь имя_модели -> модель для обучения.
        X_train_dict: словарь имя_модели -> обучающие данные для этой модели.
        X_test_dict: словарь имя_модели -> тестовые данные для этой модели.
        y_train: метки для обучающих данных.
        y_test: метки для тестовых данных.
        average: тип усреднения для recall и f1 ('binary' или 'macro'). 
                 Если None, определяется автоматически.

    Returns:
        Результаты в виде словаря:
        {model_name: {'accuracy': ..., 'recall': ..., 'f1_score': ...}, ...}
    """
    # Determine average if not provided
    if average is None:
        n_classes = len(np.unique(y_train))
        average = 'binary' if n_classes == 2 else 'macro'

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        print(f"🔧 Training {name}")
        start_time = time.time()
        # Масштабируем признаки индивидуально для каждой модели
        method = "minmax" if name == "k-NN" else "standard"
        X_tr_scaled, X_te_scaled = prepare_scaled_data(X_train_dict[name], X_test_dict[name], method=method)
        model.fit(X_tr_scaled, y_train)
        y_pred = model.predict(X_te_scaled)
        elapsed = time.time() - start_time
        print(f"⏱️ {name}: {elapsed:.2f}s")

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        print(f"{name} — Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

        results[name] = {
            'accuracy': acc,
            'recall': rec,
            'f1_score': f1
        }
    return results


# --- Utility functions for results analysis ---
def get_best_model_name(results: Dict[str, Dict[str, float]]) -> str:
    """
    Выбирает имя модели с наивысшим значением F1-score.
    """
    return max(results, key=lambda name: results[name]['f1_score'])

def show_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Отображает результаты обучения моделей в виде таблицы и сохраняет в .csv.
    """
    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "Model"

    print("\n📊 Результаты обучения моделей:\n")
    print(results_df.to_markdown())

    # Сохраняем таблицу
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/model_results.csv"
    results_df.to_csv(csv_path)
    print(f"\n💾 Таблица результатов сохранена в: {csv_path}\n")

    # Подсветка лучшей модели
    best_name = get_best_model_name(results)
    print(colored(f"🏆 Лучшая модель: {best_name}", "green", attrs=["bold"]))