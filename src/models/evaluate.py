


import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from typing import Dict, Any

def show_confusion_matrix(
    model: Any,
    X,
    y,
    output_path: str = None
):
    """
    Строит и отображает или сохраняет матрицу ошибок для модели.

    Args:
        model: обученная модель с методами predict и атрибутом classes_.
        X: признаки тестовой выборки.
        y: метки тестовой выборки.
        output_path: путь для сохранения изображения (PNG). Если None, отображает на экране.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix: {type(model).__name__}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_auc(
    models: Dict[str, Any],
    X_test_dict: Dict[str, Any],
    y_test,
    output_path: str = None
):
    """
    Строит и отображает или сохраняет ROC-кривые для набора моделей.

    Args:
        models: словарь названий моделей и обученных объектов.
        X_test_dict: словарь названий моделей и соответствующих тестовых признаков.
        y_test: метки тестовой выборки (двоичные).
        output_path: путь для сохранения изображения (PNG). Если None, отображает на экране.
    """
    if len(set(y_test)) != 2:
        print("ROC AUC доступна только для бинарной классификации.")
        return

    plt.figure()
    for name, model in models.items():
        X_test = X_test_dict.get(name)
        if X_test is None:
            continue
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_scores = model.decision_function(X_test)
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()