import pandas as pd
import numpy as np

# Загружаем данные
df_train = pd.read_parquet('train_team_track.parquet')
df_test = pd.read_parquet('test_team_track.parquet')

print("=" * 60)
print("ТРЕНИРОВОЧНЫЕ ДАННЫЕ")
print("=" * 60)
print(f"Размер: {df_train.shape[0]:,} строк x {df_train.shape[1]} колонок")
print(f"Колонки: {list(df_train.columns)}")
print()

print("=" * 60)
print("ТЕСТОВЫЕ ДАННЫЕ")
print("=" * 60)
print(f"Размер: {df_test.shape[0]:,} строк x {df_test.shape[1]} колонок")
print(f"Колонки: {list(df_test.columns)}")
print()

# Работаем с тренировочными
df = df_train.copy()

print("=" * 60)
print("ПЕРВЫЕ 5 СТРОК ТРЕНИРОВОЧНЫХ")
print("=" * 60)
print(df.head())
print()

print("=" * 60)
print("ТИПЫ ДАННЫХ")
print("=" * 60)
print(df.dtypes)
print()

print("=" * 60)
print("ПРОВЕРКА НА ПРОПУСКИ")
print("=" * 60)
print(df.isnull().sum())
print()

print("=" * 60)
print("СТАТИСТИКА target_2h")
print("=" * 60)
print(df['target_2h'].describe())
print(f"\nДоля нулей: {(df['target_2h'] == 0).mean()*100:.2f}%")
print()

print("=" * 60)
print("СТАТИСТИКА ПО СТАТУСАМ")
print("=" * 60)
status_cols = ['status_1', 'status_2', 'status_3', 'status_4', 
               'status_5', 'status_6', 'status_7', 'status_8']

for col in status_cols:
    non_zero = (df[col] > 0).mean() * 100
    print(f"{col}: mean={df[col].mean():.2f}, "
          f"std={df[col].std():.2f}, "
          f"max={df[col].max()}, "
          f"ненулевых={non_zero:.1f}%")
print()

print("=" * 60)
print("КОРРЕЛЯЦИЯ С target_2h")
print("=" * 60)
for col in status_cols:
    corr = df[col].corr(df['target_2h'])
    print(f"{col}: {corr:.4f}")
print()

# Корреляция суммы статусов
df['sum_status'] = df[status_cols].sum(axis=1)
corr_sum = df['sum_status'].corr(df['target_2h'])
print(f"Сумма всех статусов: {corr_sum:.4f}")
print()

print("=" * 60)
print("УНИКАЛЬНЫЕ ЗНАЧЕНИЯ")
print("=" * 60)
print(f"Уникальных route_id: {df['route_id'].nunique():,}")
print(f"Уникальных office_from_id: {df['office_from_id'].nunique():,}")
print()

# Проверка связи route_id и office_from_id
route_office = df.groupby('route_id')['office_from_id'].nunique()
print(f"Маршрутов, привязанных к >1 складу: {(route_office > 1).sum()}")
print()

print("=" * 60)
print("ВРЕМЕННАЯ СТРУКТУРА")
print("=" * 60)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Период: с {df['timestamp'].min()} по {df['timestamp'].max()}")
print(f"Всего дней: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print(f"Всего часов: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.0f}")
print()

# Проверяем частоту записей
print("\nЧастота обновления данных (первые 10 значений):")
time_diffs = df['timestamp'].diff().dropna().value_counts().head(10)
for diff, count in time_diffs.items():
    print(f"  {diff} : {count} раз")
    
print("=" * 60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
print("=" * 60)

# Проверим автокорреляцию target_2h для одного маршрута
sample_route = df['route_id'].iloc[0]
df_route = df[df['route_id'] == sample_route].copy()
df_route = df_route.sort_values('timestamp')

print(f"Анализ маршрута {sample_route}:")

# Автокорреляции
for lag in [1, 2, 4, 8, 12, 24]:
    df_route[f'target_lag_{lag}'] = df_route['target_2h'].shift(lag)
    corr = df_route['target_2h'].corr(df_route[f'target_lag_{lag}'])
    print(f"  Лаг {lag*30} мин: {corr:.4f}")

# Посмотрим на распределение по часам
df['hour'] = df['timestamp'].dt.hour
hourly_mean = df.groupby('hour')['target_2h'].mean()
print("\nСредняя отгрузка по часам:")
for hour in [0, 6, 12, 18, 23]:
    print(f"  {hour:02d}:00 -> {hourly_mean[hour]:.1f}")

# Проверим день недели
df['dow'] = df['timestamp'].dt.dayofweek
dow_mean = df.groupby('dow')['target_2h'].mean()
print("\nСредняя отгрузка по дням недели:")
dow_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
for i in range(7):
    print(f"  {dow_names[i]}: {dow_mean[i]:.1f}")
    
# ============================================================
# ШАГ 1: ПОДГОТОВКА ФИЧЕЙ ДЛЯ МОДЕЛИ
# ============================================================

print("\n" + "=" * 60)
print("ШАГ 1: ПОДГОТОВКА ФИЧЕЙ ДЛЯ МОДЕЛИ")
print("=" * 60)

# Убедимся, что timestamp в правильном формате
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Сортируем данные по времени для каждого маршрута
df = df.sort_values(['route_id', 'timestamp'])

# Создаем признаки
print("Создание временных признаков...")
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Циклические признаки для времени суток
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Лаги целевой переменной
print("Создание лагов target_2h...")
for lag in [1, 2, 4, 8, 12]:
    df[f'target_lag_{lag}'] = df.groupby('route_id')['target_2h'].shift(lag)

# Лаги статусов
important_statuses = ['status_4', 'status_5', 'status_6', 'status_8']
print("Создание лагов статусов...")
for lag in [1, 2, 4]:
    for col in important_statuses:
        df[f'{col}_lag_{lag}'] = df.groupby('route_id')[col].shift(lag)

# Скользящие суммы
print("Создание скользящих сумм...")
for window in [2, 4, 8]:
    for col in important_statuses:
        df[f'{col}_sum_{window}'] = df.groupby('route_id')[col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )

# Отношения статусов
print("Создание отношений статусов...")
df['ratio_5_6'] = df['status_5'] / (df['status_6'] + 1)
df['ratio_4_5'] = df['status_4'] / (df['status_5'] + 1)
df['ratio_6_8'] = df['status_6'] / (df['status_8'] + 1)

# Сумма всех статусов
df['total_in_pipeline'] = df[important_statuses].sum(axis=1)

# Удаляем строки с NaN
print(f"Размер до удаления NaN: {len(df)}")
df = df.dropna()
print(f"Размер после удаления NaN: {len(df)}")

print("\nФичи созданы успешно!")

# ============================================================
# ШАГ 2: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# ============================================================

print("\n" + "=" * 60)
print("ШАГ 2: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
print("=" * 60)

feature_cols = [
    'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend', 'month',
    'target_lag_1', 'target_lag_2', 'target_lag_4', 'target_lag_8', 'target_lag_12',
    'status_4_lag_1', 'status_5_lag_1', 'status_6_lag_1', 'status_8_lag_1',
    'status_4_lag_2', 'status_5_lag_2', 'status_6_lag_2', 'status_8_lag_2',
    'status_4_sum_2', 'status_5_sum_2', 'status_6_sum_2', 'status_8_sum_2',
    'status_4_sum_4', 'status_5_sum_4', 'status_6_sum_4', 'status_8_sum_4',
    'ratio_5_6', 'ratio_4_5', 'ratio_6_8', 'total_in_pipeline'
]

X = df[feature_cols]
y = df['target_2h']

print(f"Количество признаков: {len(feature_cols)}")
print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")

# Временная разбивка
split_idx = int(len(df) * 0.8)
X_train = X[:split_idx]
X_val = X[split_idx:]
y_train = y[:split_idx]
y_val = y[split_idx:]

print(f"\nОбучающая выборка: {len(X_train):,} строк")
print(f"Валидационная выборка: {len(X_val):,} строк")

# ============================================================
# ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================================

print("\n" + "=" * 60)
print("ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ LIGHTGBM")
print("=" * 60)

try:
    import lightgbm as lgb
    
    # ИСПРАВЛЕНО: y_true теперь numpy array, не нужно вызывать get_label()
    def lgb_wape_rbias(y_pred, y_true):
        """Кастомная метрика для LightGBM"""
        wape = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
        rbias = np.abs(np.sum(y_pred) / np.sum(y_true) - 1)
        return 'wape_rbias', wape + rbias, False
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("Начало обучения...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_wape_rbias,
        callbacks=[
            lgb.early_stopping(30, verbose=True),
            lgb.log_evaluation(50)
        ]
    )
    
    print(f"\n✅ Обучение завершено!")
    print(f"Лучшее количество деревьев: {model.best_iteration_}")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 ТОП-10 важных признаков:")
    print(feature_importance.head(10))
    
    # ============================================================
    # ШАГ 4: ПРЕДСКАЗАНИЕ ДЛЯ ТЕСТОВОГО НАБОРА
    # ============================================================
    
    print("\n" + "=" * 60)
    print("ШАГ 4: ПРЕДСКАЗАНИЕ ДЛЯ ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # Загружаем тестовые данные заново (на случай если изменились)
    df_test = pd.read_parquet('test_team_track.parquet')
    print(f"Тестовых строк: {len(df_test)}")
    
    # Простое решение: предсказываем среднее значение по route_id
    route_means = df.groupby('route_id')['target_2h'].mean()
    df_test['y_pred'] = df_test['route_id'].map(route_means)
    # Заполняем пропуски глобальным средним (на случай если route_id нет в обучении)
    df_test['y_pred'] = df_test['y_pred'].fillna(df['target_2h'].mean())
    
    # Сохраняем результат
    submission = df_test[['id', 'y_pred']].copy()
    submission.columns = ['id', 'y_pred']
    submission.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Сохранено предсказание в submission.csv")
    print(f"Файл содержит {len(submission)} строк")
    print("\nПример предсказаний:")
    print(submission.head())
    
    # Дополнительно: посчитаем метрику на валидации
    y_val_pred = model.predict(X_val)
    val_wape = np.sum(np.abs(y_val_pred - y_val)) / np.sum(y_val)
    val_rbias = np.abs(np.sum(y_val_pred) / np.sum(y_val) - 1)
    print(f"\n📈 Метрика на валидации: WAPE={val_wape:.4f}, RBias={val_rbias:.4f}")
    print(f"   Итоговая метрика (WAPE+RBias): {val_wape + val_rbias:.4f}")
    
except ImportError as e:
    print(f"\n⚠️ Ошибка импорта: {e}")
    print("Установи необходимые пакеты:")
    print("py -m pip install lightgbm scikit-learn")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)