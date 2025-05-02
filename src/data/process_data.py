import pandas as pd

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет базовую предобработку:
    - Приведение концентрации к числовому типу
    - Удаление строк с пропущенными значениями в любом столбце
    - Сброс индекса после очистки
    """
    df = df.copy()
    # Преобразуем концентрацию к float
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')
    # Удаляем строки с NaN во всех столбцах
    df = df.dropna(how='any')
    # Сбрасываем индекс для чистоты
    df = df.reset_index(drop=True)
    return df


def analyze_data(df: pd.DataFrame) -> None:
    """
    Выполняет анализ данных:
    - Печать примера данных
    - Печать диапазона концентраций
    - Печать числа уникальных антибиотиков и их списка
    """
    print("---------------------------------------------------------------")
    print("Пример данных из итогового DataFrame:\n")
    print(df.head())

    range_df = df.groupby("antibiotic")["concentration"].agg(["min", "max"])
    print("\nДиапазон концентраций для каждого антибиотика:\n", range_df)

    unique_ants = df["antibiotic"].dropna().unique()
    print("---------------------------------------------------------------")
    print(f"Число уникальных антибиотиков: {len(unique_ants)}")
    print("Список антибиотиков:", unique_ants)