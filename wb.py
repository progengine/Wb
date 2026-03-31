import pandas as pd
import numpy as np

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
df_train = pd.read_parquet('train_team_track.parquet')
df_test = pd.read_parquet('test_team_track.parquet')

df = df_train.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================================
# АНАЛИЗ ДАННЫХ
# ============================================================
print("=" * 60)
print("1. АНАЛИЗ ДАННЫХ")
print("=" * 60)

print("Статистика target_2h:")
print(df['target_2h'].describe())
print(f"\nДоля нулей: {(df['target_2h'] == 0).mean()*100:.2f}%")
print()

status_cols = ['status_1', 'status_2', 'status_3', 'status_4', 
               'status_5', 'status_6', 'status_7', 'status_8']

print("Корреляция статусов с target_2h:")
for col in status_cols:
    print(f"  {col}: {df[col].corr(df['target_2h']):.4f}")
print()

print(f"Уникальных route_id: {df['route_id'].nunique():,}")
print(f"Уникальных office_from_id: {df['office_from_id'].nunique():,}")
print()

print(f"Период данных: с {df['timestamp'].min()} по {df['timestamp'].max()}")
print(f"Всего дней: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Автокорреляция
sample_route = df['route_id'].iloc[0]
df_route = df[df['route_id'] == sample_route].sort_values('timestamp')
print(f"Автокорреляция target_2h (маршрут {sample_route}):")
for lag in [1, 2, 4, 8, 12]:
    df_route[f'target_lag_{lag}'] = df_route['target_2h'].shift(lag)
    corr = df_route['target_2h'].corr(df_route[f'target_lag_{lag}'])
    print(f"  Лаг {lag*30} мин: {corr:.4f}")

# Сезонность
df['hour'] = df['timestamp'].dt.hour
hourly_mean = df.groupby('hour')['target_2h'].mean()
print("\nСредняя отгрузка по часам:")
for hour in [0, 6, 12, 18, 23]:
    print(f"  {hour:02d}:00 -> {hourly_mean[hour]:.1f}")

df['dow'] = df['timestamp'].dt.dayofweek
dow_mean = df.groupby('dow')['target_2h'].mean()
print("\nСредняя отгрузка по дням недели:")
dow_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
for i in range(7):
    print(f"  {dow_names[i]}: {dow_mean[i]:.1f}")

# ============================================================
# ПОДГОТОВКА ФИЧЕЙ
# ============================================================
print("\n" + "=" * 60)
print("2. ПОДГОТОВКА ФИЧЕЙ")
print("=" * 60)

df = df.sort_values(['route_id', 'timestamp'])

# Временные признаки
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Лаги
for lag in [1, 2, 4, 8, 12]:
    df[f'target_lag_{lag}'] = df.groupby('route_id')['target_2h'].shift(lag)

important_statuses = ['status_4', 'status_5', 'status_6', 'status_8']
for lag in [1, 2, 4]:
    for col in important_statuses:
        df[f'{col}_lag_{lag}'] = df.groupby('route_id')[col].shift(lag)

# Скользящие суммы
for window in [2, 4, 8]:
    for col in important_statuses:
        df[f'{col}_sum_{window}'] = df.groupby('route_id')[col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )

# Отношения
df['ratio_5_6'] = df['status_5'] / (df['status_6'] + 1)
df['ratio_4_5'] = df['status_4'] / (df['status_5'] + 1)
df['ratio_6_8'] = df['status_6'] / (df['status_8'] + 1)
df['total_in_pipeline'] = df[important_statuses].sum(axis=1)

df = df.dropna()
print(f"Размер после создания фичей: {len(df):,} строк")
print()

# ============================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
print("=" * 60)
print("3. ОБУЧЕНИЕ МОДЕЛИ LIGHTGBM")
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

split_idx = int(len(df) * 0.8)
X_train = X[:split_idx]
X_val = X[split_idx:]
y_train = y[:split_idx]
y_val = y[split_idx:]

print(f"Обучающая: {len(X_train):,} строк")
print(f"Валидационная: {len(X_val):,} строк")
print()

import lightgbm as lgb

def lgb_wape_rbias(y_pred, y_true):
    wape = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
    rbias = np.abs(np.sum(y_pred) / np.sum(y_true) - 1)
    return 'wape_rbias', wape + rbias, False

model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
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
    callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
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

# Метрика на валидации
y_val_pred = model.predict(X_val)
val_wape = np.sum(np.abs(y_val_pred - y_val)) / np.sum(y_val)
val_rbias = np.abs(np.sum(y_val_pred) / np.sum(y_val) - 1)
print(f"\n📈 Метрика на валидации:")
print(f"   WAPE = {val_wape:.4f}")
print(f"   RBias = {val_rbias:.4f}")
print(f"   Итог = {val_wape + val_rbias:.4f}")

# ============================================================
# ПРЕДСКАЗАНИЕ ДЛЯ ТЕСТА
# ============================================================
print("\n" + "=" * 60)
print("4. ПРЕДСКАЗАНИЕ ДЛЯ ТЕСТОВЫХ ДАННЫХ")
print("=" * 60)

df_test = pd.read_parquet('test_team_track.parquet')
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

# Последние известные значения
last_statuses = df.groupby('route_id')[important_statuses].last().reset_index()
last_targets = df.groupby('route_id')['target_2h'].last().reset_index()
last_targets.columns = ['route_id', 'last_target']

# Временные признаки
df_test['hour'] = df_test['timestamp'].dt.hour
df_test['dayofweek'] = df_test['timestamp'].dt.dayofweek
df_test['month'] = df_test['timestamp'].dt.month
df_test['is_weekend'] = (df_test['dayofweek'] >= 5).astype(int)
df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)

# Присоединяем исторические данные
df_test = df_test.merge(last_statuses, on='route_id', how='left')
df_test = df_test.merge(last_targets, on='route_id', how='left')

# Заполняем пропуски
for col in important_statuses:
    df_test[col] = df_test[col].fillna(df[col].median())
df_test['last_target'] = df_test['last_target'].fillna(df['target_2h'].mean())

# Создаём признаки для теста
for lag in [1, 2, 4]:
    for col in important_statuses:
        df_test[f'{col}_lag_{lag}'] = df_test[col]

for window in [2, 4, 8]:
    for col in important_statuses:
        df_test[f'{col}_sum_{window}'] = df_test[col] * window

df_test['ratio_5_6'] = df_test['status_5'] / (df_test['status_6'] + 1)
df_test['ratio_4_5'] = df_test['status_4'] / (df_test['status_5'] + 1)
df_test['ratio_6_8'] = df_test['status_6'] / (df_test['status_8'] + 1)
df_test['total_in_pipeline'] = df_test[important_statuses].sum(axis=1)

for lag in [1, 2, 4, 8, 12]:
    df_test[f'target_lag_{lag}'] = df_test['last_target']

# Коэффициент времени суток
hour_multiplier = df.groupby('hour')['target_2h'].mean()
hour_multiplier = hour_multiplier / hour_multiplier.mean()
df_test['hour_multiplier'] = df_test['hour'].map(hour_multiplier).fillna(1.0)

# Предсказание
available_features = [col for col in feature_cols if col in df_test.columns]
X_test = df_test[available_features].fillna(0)
y_pred_base = model.predict(X_test)
df_test['y_pred'] = y_pred_base * df_test['hour_multiplier']
df_test['y_pred'] = df_test['y_pred'].clip(lower=0, upper=df['target_2h'].quantile(0.99))

# Сохраняем результат
submission = df_test[['id', 'y_pred']].copy()
submission.columns = ['id', 'y_pred']
submission.to_csv('submission.csv', index=False)

print(f"\n✅ Сохранено в submission.csv")
print(f"Файл содержит {len(submission)} строк")
print(f"Test pred mean: {submission['y_pred'].mean():.2f}")
print(f"Train target mean: {df['target_2h'].mean():.2f}")
print("\n📋 Пример предсказаний (первые 10):")
print(submission.head(10))

print("\n" + "=" * 60)
print(" ГОТОВО!")
print("=" * 60)