


import os
import random
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(
    counts: pd.Series,
    output_path: Union[str, Path] = None
):
    """
    Строит и отображает или сохраняет гистограмму распределения категорий.

    Args:
        counts: Series, где индекс — категории, а значения — количество.
        output_path: Путь к файлу PNG для сохранения. Если None, график отображается.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index, counts.values)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            int(bar.get_height()),
            ha='center'
        )
    plt.xlabel('Категория')
    plt.ylabel('Количество')
    plt.title('Распределение категорий')
    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_va_graphs(
    graph_data: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path]
):
    """
    Строит и сохраняет графики вольтамперных характеристик для каждого антибиотика.

    Args:
        graph_data: Словарь, где ключ — название антибиотика,
            значение — dict с ключами 'voltages' (array-like) и 'currents' (list of array-like).
        output_dir: Папка для сохранения PNG-файлов.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for antibiotic, data in graph_data.items():
        voltages = data.get('voltages')
        currents_list = data.get('currents', [])
        if voltages is None or not currents_list:
            continue

        currents = random.choice(currents_list)
        plt.figure(figsize=(8, 6))
        plt.plot(voltages, currents)
        plt.title(f"Вольтамперная характеристика для {antibiotic}")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.tight_layout()

        file_path = out_dir / f"{antibiotic}_VA.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()