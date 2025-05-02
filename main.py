#!/usr/bin/env python
import os
import argparse
import pandas as pd
from zipfile import ZipFile
from pathlib import Path

from src.data.load_data import extract_data, load_data
from src.data.process_data import process_data, analyze_data
from src.data.split_data import split_data
from src.features.select_feature import prepare_scaled_data
from src.models.train import get_models, train_and_evaluate, tune_model, show_results_table, get_best_model_name
from src.models.evaluate import show_confusion_matrix, plot_roc_auc
from src.visualizations.plot_results import plot_histogram, plot_va_graphs

RAW_DIR = 'data/raw/'
PROCESSED_CSV = 'data/processed/processed_data.csv'
VA_GRAPH_DIR = 'reports/figures/VA_graphs'
HISTOGRAM_PATH = 'reports/figures/antibiotics_distribution.png'

def step_collect(args):
    # 1) Распаковка архива
    zip_path = args.archive or os.path.join(RAW_DIR, 'csv_ivium_new_data.zip')
    if os.path.exists(zip_path):
        print("📦 Распаковываем архив...")
        extract_data(zip_path, RAW_DIR)
    # 2) Сбор и предобработка
    df = load_data(RAW_DIR)
    df = process_data(df)
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"💾 Собранный CSV сохранён: {PROCESSED_CSV}")

def step_analyze(args):
    # 3) Анализ базы данных
    df = pd.read_csv(PROCESSED_CSV)
    analyze_data(df)
    os.makedirs(os.path.dirname(HISTOGRAM_PATH), exist_ok=True)
    plot_histogram(df['antibiotic'].value_counts(), output_path=HISTOGRAM_PATH)
    print(f"💾 Гистограмма сохранена: {HISTOGRAM_PATH}")
    os.makedirs(VA_GRAPH_DIR, exist_ok=True)
    graph_data = {}
    voltage_cols = [col for col in df.columns if col not in ('antibiotic','concentration')]
    voltages = sorted([float(v) for v in voltage_cols])
    keys = [f"{v:.3f}" for v in voltages]
    for _, row in df.iterrows():
        ant = row['antibiotic']
        currents = [row[k] for k in keys]
        graph_data.setdefault(ant, {'voltages': voltages, 'currents': []})['currents'].append(currents)
    plot_va_graphs(graph_data, VA_GRAPH_DIR)
    print(f"💾 VA-графики сохранены в: {VA_GRAPH_DIR}")

def step_train_binary(args):
    # 4) Обучение бинарной классификации
    import time
    start = time.time()

    # Удаление старых графиков confusion_matrix и ROC
    import glob
    fig_dir = os.path.join('reports', 'figures')
    patterns = ['confusion_binary_*.png', 'roc_binary_*.png']
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(fig_dir, pattern)):
            try:
                os.remove(file_path)
            except OSError:
                pass

    # Загрузка и подготовка данных
    df = pd.read_csv(PROCESSED_CSV)
    X = df.drop(['antibiotic', 'concentration'], axis=1)
    y = (df['antibiotic'] != 'milk').astype(int)

    # Сплит
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Создаем словари сырых данных для train_and_evaluate
    raw_train = {name: X_train for name in get_models({}, {}, {}, {})}
    raw_test  = {name: X_test  for name in get_models({}, {}, {}, {})}

    # Создаем модели
    models = get_models({}, {}, {}, {})

    # Тюнинг гиперпараметров, если нужно
    if args.tune:
        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10]},
            'k-NN': {'n_neighbors': list(range(1, 21)), 'weights': ['uniform', 'distance'], 'p': [1, 2]},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'Gradient Boosting': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
        }
        for name, model in models.items():
            grid = param_grids.get(name)
            if not grid:
                continue
            # Выбираем метод масштабирования для тюнинга
            method = "minmax" if name == "k-NN" else "standard"
            X_tr_scaled, _ = prepare_scaled_data(X_train, X_test, method=method)
            best_params = tune_model(
                model,
                grid,
                X_tr_scaled,
                y_train,
                average='binary'
            )
            model.set_params(**best_params)
            models[name] = model

    # Обучаем и оцениваем все модели
    results = train_and_evaluate(
        models,
        raw_train,
        raw_test,
        y_train,
        y_test
    )

    # Выводим результаты
    show_results_table(results)
    best_name = get_best_model_name(results)
    print(f"🏆 Лучшая модель: {best_name}")

    # Матрица ошибок и ROC для лучшей модели
    # Снова масштабируем для лучшей модели
    method = "minmax" if best_name == "k-NN" else "standard"
    _, X_te_best = prepare_scaled_data(X_train, X_test, method=method)
    show_confusion_matrix(
        models[best_name],
        X_te_best,
        y_test,
        output_path=f'reports/figures/confusion_binary_{best_name.replace(" ", "_")}.png'
    )
    plot_roc_auc(
        {best_name: models[best_name]},
        {best_name: X_te_best},
        y_test,
        output_path=f'reports/figures/roc_binary_{best_name.replace(" ", "_")}.png'
    )

    end = time.time()
    print(f"✅ Полное время выполнения: {end - start:.2f} сек")

def step_train_multiclass(args):
    # 5) Обучение многоклассовой классификации
    import time
    start = time.time()
    df = pd.read_csv(PROCESSED_CSV)
    X = df.drop(['antibiotic','concentration'], axis=1)
    y = df['antibiotic']
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    X_tr, X_te = prepare_scaled_data(X_train, X_test)
    models = get_models({}, {}, {}, {})
    if args.tune:
        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10]},
            'k-NN': {'n_neighbors': list(range(1, 21)), 'weights': ['uniform', 'distance'], 'p': [1, 2]},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'Gradient Boosting': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
        }
        for name, model in models.items():
            grid = param_grids.get(name)
            if not grid:
                continue
            best_params = tune_model(
                model,
                grid,
                X_tr,
                y_train,
                average='macro'
            )
            model.set_params(**best_params)
            models[name] = model
    results = train_and_evaluate(
        models,
        {name: X_tr for name in models},
        {name: X_te for name in models},
        y_train, y_test
    )
    show_results_table(results)
    best_name = get_best_model_name(results)
    print(f"🏆 Лучшая модель: {best_name}")
    print("🎯 Результаты мультиклассовой классификации:\n", results)
    show_confusion_matrix(models['Logistic Regression'], X_te, y_test, output_path='reports/figures/confusion_multi.png')
    plot_roc_auc(models, {name: X_te for name in models}, y_test, output_path='reports/figures/roc_multi.png')
    end = time.time()
    print(f"✅ Полное время выполнения: {end - start:.2f} сек")

def main():
    parser = argparse.ArgumentParser(description="Milkyway Classification Pipeline")
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('collect', help='Сбор и сохранение данных')
    sub.add_parser('analyze', help='Анализ базы данных')
    tb = sub.add_parser('train_binary', help='Бинарная классификация')
    tb.add_argument('--tune', action='store_true', help='Подбирать гиперпараметры')
    tb.add_argument('--test-size', type=float, default=0.2)
    tb.add_argument('--random-state', type=int, default=42)
    tm = sub.add_parser('train_multiclass', help='Многоклассовая классификация')
    tm.add_argument('--tune', action='store_true', help='Подбирать гиперпараметры')
    tm.add_argument('--test-size', type=float, default=0.2)
    tm.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--archive', type=str, help='Путь к .zip архиву с данными')
    args = parser.parse_args()

    if args.command == 'collect':
        step_collect(args)
    elif args.command == 'analyze':
        step_analyze(args)
    elif args.command == 'train_binary':
        step_train_binary(args)
    elif args.command == 'train_multiclass':
        step_train_multiclass(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()