 

from zipfile import ZipFile
from pathlib import Path
import pandas as pd
import numpy as np

def extract_data(archive_path: str, extract_to: str):
    """
    Извлекает все CSV-файлы из архива в указанную папку.
    """
    with ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Считывает все CSV-файлы из папки и возвращает единый DataFrame:
      - Колонки: напряжения (Voltage_V), форматированы до 3 знаков
      - Строки: значения тока (Current_A) как float
      - Дополнительные колонки: 'antibiotic' и 'concentration' (float)
    """
    data_path = Path(data_dir)
    # Найти все CSV (исключая скрытые файлы macOS)
    csv_files = [f for f in data_path.glob("**/*.csv") if not f.name.startswith("._")]
    records = []
    voltages = None
    voltage_keys = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        # Упростить имена колонок
        df.columns = df.columns.str.replace(r'[,\s]+', '_', regex=True)

        if 'Voltage_V' in df.columns and 'Current_A' in df.columns:
            # Инициализировать список напряжений и ключей единожды
            if voltages is None:
                voltages = np.round(df['Voltage_V'].values, 3)
                voltage_keys = [f"{v:.3f}" for v in voltages]

            currents = df['Current_A'].values.astype(float)
            # Разбор имени файла для антибиотика и концентрации
            parts = csv_file.stem.split('_')
            antibiotic = parts[0] if len(parts) >= 2 else None
            try:
                concentration_val = float(parts[1])
            except (IndexError, ValueError):
                concentration_val = None

            # Собрать запись
            rec = {voltage_keys[i]: currents[i] for i in range(len(voltage_keys))}
            rec['antibiotic'] = antibiotic
            rec['concentration'] = concentration_val
            records.append(rec)

    # Если данных нет, вернуть пустой DataFrame
    if not records:
        return pd.DataFrame()

    df_all = pd.DataFrame(records)
    # Гарантировать числовой тип концентрации
    df_all['concentration'] = pd.to_numeric(df_all['concentration'], errors='coerce')
    return df_all