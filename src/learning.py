# learning4.py - Инструмент для стратифицированного разбиения датасета и обучения моделей CatBoost
import gradio as gr

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from optuna import create_study

import traceback

from catboost import CatBoostRegressor
import shap

import sys
import tempfile
from io import StringIO
import math
import re
import os

import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ========== Глобальные переменные ==========
trained_models = {}  # Словарь для хранения обученных моделей
shap_explanations = {}  # Словарь для хранения SHAP объяснений
predictions_cache = {}
log_capture = StringIO()


# ========== Функции для визуализации ==========

def plot_categorical_comparison(df1, df2, categorical_columns):
    """Строит круговые диаграммы для категориальных признаков"""
    if len(categorical_columns) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет категориальных признаков для отображения",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    n_cols = 2
    n_rows = len(categorical_columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(categorical_columns):
        first_counts = df1[col].value_counts()
        axes[i, 0].pie(first_counts.values, labels=first_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i, 0].set_title(f'{col} (Первая выборка)\nn={len(df1)}')

        second_counts = df2[col].value_counts()
        axes[i, 1].pie(second_counts.values, labels=second_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i, 1].set_title(f'{col} (Вторая выборка)\nn={len(df2)}')

    plt.tight_layout()
    return fig


def plot_numerical_comparison(df1, df2, numerical_columns):
    """Строит гистограммы для числовых признаков"""
    if len(numerical_columns) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет числовых признаков для отображения",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    n_rows = len(numerical_columns)

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(numerical_columns):
        axes[i, 0].hist(df1[col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[i, 0].axvline(df1[col].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Ср.: {df1[col].mean():.3f}')
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel('Частота')
        axes[i, 0].set_title(f'{col} (Первая выборка)\nn={len(df1)}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].hist(df2[col].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[i, 1].axvline(df2[col].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Ср.: {df2[col].mean():.3f}')
        axes[i, 1].set_xlabel(col)
        axes[i, 1].set_ylabel('Частота')
        axes[i, 1].set_title(f'{col} (Вторая выборка)\nn={len(df2)}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_visualizations(df1, df2):
    """Создаёт визуализации для категориальных и числовых признаков"""
    if df1 is None or df2 is None:
        return None, None, "❌ Сначала выполните разбиение данных!"

    categorical_cols = df1.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'cluster' in df1.columns and 'cluster' not in categorical_cols:
        categorical_cols.append('cluster')

    numerical_cols = df1.select_dtypes(include=[np.number]).columns.tolist()

    fig_cat = plot_categorical_comparison(df1, df2, categorical_cols)
    fig_num = plot_numerical_comparison(df1, df2, numerical_cols)

    return fig_cat, fig_num, f"✅ Графики построены! Категориальных: {len(categorical_cols)}, Числовых: {len(numerical_cols)}"


# ========== Функции для стратифицированного разбиения ==========

def get_datasets(dataset, column):
    """Разбивает датасет на поддатасеты по уникальным значениям указанной колонки"""
    if dataset is None:
        return None, "❌ Сначала загрузите данные!"

    if column not in dataset.columns:
        return None, f"❌ Колонка '{column}' не найдена в датасете!"

    unique_values = dataset[column].unique()
    unique_values = list(unique_values)
    unique_values.sort()

    datasets = []
    for value in unique_values:
        subset = dataset[dataset[column] == value].copy()
        subset = subset.drop(columns=[column])
        datasets.append(subset)

    return datasets, f"✅ Создано {len(datasets)} поддатасетов по колонке '{column}'"


def remove_singles(dataset, stratified_column):
    """Удаляет категории, встречающиеся только один раз"""
    few = dataset[stratified_column].value_counts()
    categories = list(few[few == 1].index)
    singles = dataset[dataset[stratified_column].isin(categories)]
    dataset = dataset.drop(singles.index)
    return dataset, singles


def make_datasets(datasets, stratified_column='population', second_size=0.5, random_state=42):
    """Из списка поддатасетов формирует две выборки"""
    if datasets is None or len(datasets) == 0:
        return None, None, "❌ Нет данных для разбиения!"

    first_datasets = []
    second_datasets = []

    for dataset in datasets:
        if stratified_column not in dataset.columns:
            first_datasets.append(dataset)
            continue

        dataset, singles = remove_singles(dataset, stratified_column)

        if len(singles) > 0:
            first_datasets.append(singles)

        if len(dataset) == 0:
            continue

        stratify = None
        unique_count = dataset[stratified_column].nunique()

        if unique_count > 1 and int(dataset.shape[0] * second_size) >= unique_count:
            stratify = dataset[stratified_column]

        first_size = 1.0 - second_size
        if int(dataset.shape[0] * first_size) == 0:
            if first_datasets:
                first_datasets[-1] = pd.concat([first_datasets[-1], dataset])
            else:
                first_datasets.append(dataset)
        else:
            local_first, local_second = train_test_split(
                dataset,
                test_size=second_size,
                stratify=stratify,
                random_state=random_state
            )
            first_datasets.append(local_first)
            second_datasets.append(local_second)

    df1 = pd.concat(first_datasets) if first_datasets else pd.DataFrame()
    df2 = pd.concat(second_datasets) if second_datasets else pd.DataFrame()

    return df1, df2, f"✅ Разбиение выполнено!\nПервая выборка: {len(df1)} строк\nВторая выборка: {len(df2)} строк"


def split_data(
        file,
        group_column,
        stratify_column,
        second_size,
        target_column,
        random_state
):
    """Основная функция: загружает данные, разбивает на поддатасеты,
    формирует первую и вторую выборки, выделяет признаки и целевую переменную"""
    if file is None:
        return (
            None, None, None, None, None, None, None,
            "❌ Загрузите CSV файл!",
            "❌", "❌", "❌"
        )

    try:
        df = pd.read_csv(file.name, index_col=0)
    except Exception as e:
        return (
            None, None, None, None, None, None, None,
            f"❌ Ошибка при загрузке файла: {str(e)}",
            "❌", "❌", "❌"
        )

    # Очищаем кэш моделей и SHAP объяснений при загрузке новых данных
    trained_models.clear()
    shap_explanations.clear()
    predictions_cache.clear()

    # Преобразуем cluster в категориальный тип
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category')

    if target_column not in df.columns:
        return (
            None, None, None, None, None, None, None,
            f"❌ Целевая переменная '{target_column}' не найдена в датасете!",
            "❌", "❌", "❌"
        )

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return (
            None, None, None, None, None, None, None,
            f"❌ Целевая переменная '{target_column}' должна быть числовой!",
            "❌", "❌", "❌"
        )

    if group_column not in df.columns:
        return (
            None, None, None, None, None, None, None,
            f"❌ Колонка для группировки '{group_column}' не найдена!",
            "❌", "❌", "❌"
        )

    datasets, msg = get_datasets(df, group_column)
    if datasets is None:
        return (
            None, None, None, None, None, None, None,
            msg,
            "❌", "❌", "❌"
        )

    df1, df2, split_msg = make_datasets(
        datasets,
        stratified_column=stratify_column,
        second_size=second_size,
        random_state=int(random_state)
    )

    if df1 is None or len(df1) == 0:
        return (
            None, None, None, None, None, None, None,
            split_msg,
            "❌", "❌", "❌"
        )

    X1 = df1.drop(columns=[target_column])
    y1 = df1[target_column]

    X2 = df2.drop(columns=[target_column])
    y2 = df2[target_column]

    # Создаём словарь с данными для визуализаций
    data_dict = {
        'X1': X1,
        'X2': X2,
        'y1': y1,
        'y2': y2,
        'cluster_col': 'cluster'
    }

    stats = f"📊 СТАТИСТИКА РАЗБИЕНИЯ\n"
    stats += f"{'=' * 50}\n\n"
    stats += f"📁 Исходный датасет: {len(df)} строк, {len(df.columns)} колонок\n"
    stats += f"📂 Группировка по: '{group_column}' (уникальных значений: {df[group_column].nunique()})\n"
    stats += f"📊 Стратификация по: '{stratify_column}'\n\n"
    stats += f"🎯 Целевая переменная: '{target_column}'\n"
    stats += f"   Тип: {df[target_column].dtype}\n"
    stats += f"   Диапазон: [{df[target_column].min():.4f}, {df[target_column].max():.4f}]\n\n"
    stats += f"📚 ПЕРВАЯ ВЫБОРКА:\n"
    stats += f"   Строк: {len(X1)}\n"
    stats += f"   Признаков: {len(X1.columns)}\n"
    stats += f"   Целевая: {y1.name}\n\n"
    stats += f"🧪 ВТОРАЯ ВЫБОРКА:\n"
    stats += f"   Строк: {len(X2)}\n"
    stats += f"   Признаков: {len(X2.columns)}\n"
    stats += f"   Целевая: {y2.name}\n\n"

    stats += f"📈 РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:\n"
    stats += f"   Первая - среднее: {y1.mean():.4f}, std: {y1.std():.4f}\n"
    stats += f"   Вторая - среднее: {y2.mean():.4f}, std: {y2.std():.4f}\n"

    if stratify_column in df1.columns:
        stats += f"\n📊 РАСПРЕДЕЛЕНИЕ ПО '{stratify_column}':\n"
        first_dist = df1[stratify_column].value_counts(normalize=True)
        second_dist = df2[stratify_column].value_counts(normalize=True)

        for cat in sorted(set(first_dist.index) | set(second_dist.index)):
            first_pct = first_dist.get(cat, 0) * 100
            second_pct = second_dist.get(cat, 0) * 100
            stats += f"   {cat}: первая {first_pct:.1f}%, вторая {second_pct:.1f}%\n"

    status = f"✅ Готово! Первая: {len(X1)}, Вторая: {len(X2)}"

    y1_df = pd.DataFrame({target_column: y1.values}, index=y1.index)
    y2_df = pd.DataFrame({target_column: y2.values}, index=y2.index)

    return (
        X1, y1_df, X2, y2_df, df1, df2, data_dict,
        stats, status,
        f"✅ Первая: {len(X1)} строк, {len(X1.columns)} признаков",
        f"✅ Вторая: {len(X2)} строк, {len(X2.columns)} признаков"
    )


# ========== Функции для обучения моделей ==========

def get_categorical_columns(df):
    """Возвращает список категориальных колонок"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'cluster' in df.columns and 'cluster' not in categorical_cols:
        categorical_cols.append('cluster')
    return categorical_cols


def train_models_for_cluster(
        X1, y1, X2, y2,
        cluster,
        n_trials,
        cv,
        direction,
        scoring,
        n_estimators_min, n_estimators_max, n_estimators_log,
        depth_min, depth_max,
        l2_leaf_reg_min, l2_leaf_reg_max, l2_leaf_reg_log,
        learning_rate_min, learning_rate_max, learning_rate_log,
        min_data_in_leaf_min, min_data_in_leaf_max,
        random_state,
        log_callback=None
):
    """Обучает две модели для указанного кластера"""
    if X1 is None or X2 is None:
        return f"❌ Сначала выполните разбиение данных!", None, None, f"❌ Нет данных!", 0, 0, ""

    cluster_col = 'cluster'

    if cluster_col not in X1.columns or cluster_col not in X2.columns:
        return f"❌ Колонка '{cluster_col}' не найдена в данных!", None, None, f"❌ Нет колонки cluster!", 0, 0, ""

    try:
        idx1 = X1[X1[cluster_col].astype(str) == str(cluster)].index
        idx2 = X2[X2[cluster_col].astype(str) == str(cluster)].index
    except Exception as e:
        return f"❌ Ошибка при фильтрации кластера {cluster}: {str(e)}", None, None, f"❌ Ошибка!", 0, 0, ""

    if len(idx1) == 0:
        return f"❌ Кластер {cluster} не найден в первой выборке! Доступные кластеры: {X1[cluster_col].unique()}", None, None, f"❌ Нет данных!", 0, 0, ""
    if len(idx2) == 0:
        return f"❌ Кластер {cluster} не найден во второй выборке! Доступные кластеры: {X2[cluster_col].unique()}", None, None, f"❌ Нет данных!", 0, 0, ""

    X1_cluster = X1.loc[idx1].drop(columns=[cluster_col])
    y1_cluster = y1.loc[idx1]

    X2_cluster = X2.loc[idx2].drop(columns=[cluster_col])
    y2_cluster = y2.loc[idx2]

    if isinstance(y1_cluster, pd.DataFrame):
        y1_cluster = y1_cluster.iloc[:, 0]
    if isinstance(y2_cluster, pd.DataFrame):
        y2_cluster = y2_cluster.iloc[:, 0]

    y1_cluster = y1_cluster.values.ravel()
    y2_cluster = y2_cluster.values.ravel()

    cat_features = get_categorical_columns(X1_cluster)

    results = []
    models = []
    best_params_list = []
    all_logs = []

    for model_num, (X_train, y_train, X_pred, y_pred, model_name) in enumerate([
        (X1_cluster, y1_cluster, X2_cluster, y2_cluster,
         f"Модель 1: {cluster} (обучена на первой, предсказывает для второй)"),
        (X2_cluster, y2_cluster, X1_cluster, y1_cluster,
         f"Модель 2: {cluster} (обучена на второй, предсказывает для первой)")
    ], 1):
        try:
            study = create_study(direction=direction, study_name=f'CatBoost_{cluster}_model_{model_num}')

            trial_logs = []

            def trial_callback(study, trial):
                raw_value = trial.value
                if scoring.startswith('neg_') and direction == 'maximize':
                    display_value = -raw_value
                    metric_name = scoring.replace('neg_', '')
                else:
                    display_value = raw_value
                    metric_name = scoring

                is_best = trial.number == study.best_trial.number

                log_line = f"Попытка {trial.number}: {metric_name} = {display_value:.6f}"
                if is_best:
                    log_line += f" ✨ (ЛУЧШАЯ)"
                log_line += f"\n  Параметры: {trial.params}\n"
                log_line += "-" * 50
                trial_logs.append(log_line)

                if (trial.number + 1) % 5 == 0 or trial.number == int(n_trials) - 1:
                    best_raw_value = study.best_value
                    if scoring.startswith('neg_') and direction == 'maximize':
                        best_display_value = -best_raw_value
                        best_metric_name = scoring.replace('neg_', '')
                    else:
                        best_display_value = best_raw_value
                        best_metric_name = scoring

                    summary = f"\n{'=' * 60}\n📊 ПРОМЕЖУТОЧНЫЙ ИТОГ (после {trial.number + 1} попыток)\n"
                    summary += f"🏆 Лучшая итерация: #{study.best_trial.number}\n"
                    summary += f"📈 Лучшее значение {best_metric_name}: {best_display_value:.6f}\n"
                    summary += f"{'=' * 60}\n"

                    current_logs = summary + "\n".join(trial_logs[-50:])
                    if log_callback:
                        log_callback(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}\n{current_logs}")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', int(n_estimators_min), int(n_estimators_max),
                                                      log=n_estimators_log),
                    'depth': trial.suggest_int('depth', int(depth_min), int(depth_max)),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', l2_leaf_reg_min, l2_leaf_reg_max,
                                                       log=l2_leaf_reg_log),
                    'learning_rate': trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max,
                                                         log=learning_rate_log),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', int(min_data_in_leaf_min),
                                                          int(min_data_in_leaf_max)),
                    'verbose': False,
                    'random_seed': int(random_state)
                }

                model = CatBoostRegressor(**params)

                try:
                    scores = cross_val_score(model, X_train, y_train, cv=int(cv), scoring=scoring,
                                             params={'cat_features': cat_features})
                    return scores.mean()
                except Exception as e:
                    if direction == 'maximize':
                        return -1e10
                    else:
                        return 1e10

            study.optimize(objective, n_trials=int(n_trials), callbacks=[trial_callback], show_progress_bar=True)

            best_params = study.best_params
            best_trial = study.best_trial

            best_raw_value = study.best_value
            if scoring.startswith('neg_') and direction == 'maximize':
                best_display_value = -best_raw_value
                best_metric_name = scoring.replace('neg_', '')
            else:
                best_display_value = best_raw_value
                best_metric_name = scoring

            best_model = CatBoostRegressor(
                **best_params,
                cat_features=cat_features,
                verbose=False,
                random_seed=int(random_state)
            )
            best_model.fit(X_train, y_train, cat_features=cat_features)

            predictions = best_model.predict(X_pred)

            final_summary = f"\n{'=' * 60}\n🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ\n"
            final_summary += f"✅ Лучшая итерация: #{best_trial.number}\n"
            final_summary += f"📈 Лучшее значение CV ({best_metric_name}): {best_display_value:.6f}\n"
            final_summary += f"{'=' * 60}\n"

            model_logs = final_summary + "\n".join(trial_logs)
            all_logs.append(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}\n{model_logs}")

            mse = mean_squared_error(y_pred, predictions)
            mae = mean_absolute_error(y_pred, predictions)
            r2 = r2_score(y_pred, predictions)

            result_text = f"📊 {model_name}\n"
            result_text += f"{'=' * 60}\n"
            result_text += f"✅ Лучшие параметры (итерация #{best_trial.number}):\n"
            for key, value in best_params.items():
                result_text += f"   {key}: {value}\n"
            result_text += f"\n📈 Лучшее значение CV ({best_metric_name}): {best_display_value:.6f}\n"
            result_text += f"\n📊 МЕТРИКИ НА ТЕСТОВЫХ ДАННЫХ:\n"
            result_text += f"   📉 MSE (среднеквадратичная ошибка): {mse:.6f}\n"
            result_text += f"   📊 MAE (средняя абсолютная ошибка): {mae:.6f}\n"
            result_text += f"   📈 R² (коэффициент детерминации): {r2:.6f}\n"
            result_text += f"\n🔢 Количество попыток: {int(n_trials)}\n"
            result_text += f"🔢 CV фолдов: {int(cv)}\n"

            results.append(result_text)
            models.append(best_model)
            best_params_list.append(best_params)

        except Exception as e:
            error_msg = f"❌ Ошибка при обучении {model_name}: {str(e)}\n{traceback.format_exc()}"
            results.append(error_msg)
            models.append(None)
            best_params_list.append(None)
            all_logs.append(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}\nОшибка: {str(e)}")

    trained_models[str(cluster)] = {
        'model1': models[0] if len(models) > 0 else None,
        'model2': models[1] if len(models) > 1 else None,
        'params1': best_params_list[0] if len(best_params_list) > 0 else None,
        'params2': best_params_list[1] if len(best_params_list) > 1 else None,
        'X1_size': len(X1_cluster),
        'X2_size': len(X2_cluster)
    }

    # Очищаем SHAP объяснения для этого кластера
    shap_key1 = f"{cluster}_model1"
    shap_key2 = f"{cluster}_model2"
    if shap_key1 in shap_explanations:
        del shap_explanations[shap_key1]
    if shap_key2 in shap_explanations:
        del shap_explanations[shap_key2]

    full_result = "\n\n".join(results)
    full_logs = "\n".join(all_logs)

    if log_callback:
        log_callback(full_logs)

    if models[0] and models[1]:
        try:
            preds1 = models[0].predict(X2_cluster)
            preds2 = models[1].predict(X1_cluster)

            predictions_cache[str(cluster)] = {
                'preds1': preds1,
                'preds2': preds2
            }

            y2_values = y2_cluster if isinstance(y2_cluster, np.ndarray) else np.array(y2_cluster)
            y1_values = y1_cluster if isinstance(y1_cluster, np.ndarray) else np.array(y1_cluster)

            preds_df1 = pd.DataFrame({
                'Факт (y2)': y2_values[:10],
                'Предсказание (модель 1)': preds1[:10],
                'Ошибка': y2_values[:10] - preds1[:10]
            })

            preds_df2 = pd.DataFrame({
                'Факт (y1)': y1_values[:10],
                'Предсказание (модель 2)': preds2[:10],
                'Ошибка': y1_values[:10] - preds2[:10]
            })

            return full_result, preds_df1, preds_df2, f"✅ Модели для кластера {cluster} обучены и сохранены в кэш!", len(
                X1_cluster), len(X2_cluster), full_logs
        except Exception as e:
            return full_result, None, None, f"⚠️ Ошибка при создании превью прогнозов: {str(e)}", len(X1_cluster), len(
                X2_cluster), full_logs
    else:
        return full_result, None, None, f"⚠️ Модели для кластера {cluster} обучены с ошибками", len(X1_cluster), len(
            X2_cluster), full_logs


def get_available_clusters(X1):
    """Возвращает список доступных кластеров"""
    if X1 is None:
        return []
    if 'cluster' not in X1.columns:
        return []
    clusters = X1['cluster'].unique()
    return sorted(clusters)


def show_cached_models():
    """Показывает информацию о всех обученных моделях в кэше"""
    if not trained_models:
        return "📭 Кэш пуст. Сначала обучите модели для какого-либо кластера."

    result = "🗂️ Кэшированные модели:\n\n"
    for i, (cluster, data) in enumerate(trained_models.items(), 1):
        result += f"{i}. Кластер {cluster}\n"
        result += f"   Модель 1 (первая→вторая): {'есть' if data.get('model1') else 'нет'}\n"
        result += f"   Модель 2 (вторая→первая): {'есть' if data.get('model2') else 'нет'}\n"
        result += f"   Размер первой выборки: {data.get('X1_size', 'N/A')}\n"
        result += f"   Размер второй выборки: {data.get('X2_size', 'N/A')}\n"
        if data.get('params1'):
            result += f"   Параметры модели 1: {data['params1']}\n"
        if data.get('params2'):
            result += f"   Параметры модели 2: {data['params2']}\n"
        result += "\n"

    result += f"\n📊 Всего кластеров в кэше: {len(trained_models)}"
    return result


def clear_cache():
    """Очищает кэш моделей и SHAP объяснений"""
    trained_models.clear()
    shap_explanations.clear()
    predictions_cache.clear()
    return "🗑️ Кэш моделей и SHAP объяснений очищен!"


# ========== Функции для SHAP визуализаций ==========

def get_shap_explained(model, X, indices):
    """Возвращает SHAP объяснения для CatBoost модели"""
    feature_names = list(X.columns)
    X_for_shap = X.iloc[indices].values

    explainer = shap.TreeExplainer(model)
    explained = explainer.shap_values(X_for_shap)

    if isinstance(explained, list):
        explained = np.array(explained)

    if hasattr(shap, 'Explanation'):
        explained_obj = shap.Explanation(
            values=explained,
            base_values=explainer.expected_value,
            data=X_for_shap,
            feature_names=feature_names
        )
    else:
        explained_obj = type('obj', (object,), {
            'values': explained,
            'base_values': explainer.expected_value,
            'feature_names': feature_names
        })

    return explained_obj


def create_shap_plots_for_cluster(data_dict, cluster, plot_type, summary_plot_type='dot'):
    """Создаёт SHAP графики для обеих моделей кластера"""
    if data_dict is None:
        return None, None, "❌ Сначала выполните разбиение данных!"

    # cluster уже должен быть строкой
    if cluster not in trained_models:
        return None, None, f"❌ Модели для кластера {cluster} не найдены в кэше!"

    model_data = trained_models[cluster]
    X1 = data_dict.get('X1')
    X2 = data_dict.get('X2')

    if X1 is None or X2 is None:
        return None, None, "❌ Данные не найдены!"

    cluster_col = 'cluster'
    idx1 = X1[X1[cluster_col].astype(str) == str(cluster)].index
    idx2 = X2[X2[cluster_col].astype(str) == str(cluster)].index

    X1_cluster = X1.loc[idx1].drop(columns=[cluster_col])
    X2_cluster = X2.loc[idx2].drop(columns=[cluster_col])

    model1 = model_data.get('model1')
    model2 = model_data.get('model2')

    if model1 is None or model2 is None:
        return None, None, f"❌ Одна из моделей для кластера {cluster} не обучена!"

    shap_key1 = f"{cluster}_model1"
    shap_key2 = f"{cluster}_model2"

    if shap_key1 in shap_explanations:
        explained1 = shap_explanations[shap_key1]
    else:
        explained1 = get_shap_explained(model1, X2_cluster, range(len(X2_cluster)))
        shap_explanations[shap_key1] = explained1

    if shap_key2 in shap_explanations:
        explained2 = shap_explanations[shap_key2]
    else:
        explained2 = get_shap_explained(model2, X1_cluster, range(len(X1_cluster)))
        shap_explanations[shap_key2] = explained2

    if plot_type == 'summary':
        fig_left = shap_summary_plot(explained1, X2_cluster, summary_plot_type,
                                     title=f"Модель 1: обучена на {cluster} (первая→вторая)")
        fig_right = shap_summary_plot(explained2, X1_cluster, summary_plot_type,
                                      title=f"Модель 2: обучена на {cluster} (вторая→первая)")
    elif plot_type == 'decision':
        fig_left = shap_decision_plot(explained1, X2_cluster,
                                      title=f"Модель 1: обучена на {cluster} (первая→вторая)")
        fig_right = shap_decision_plot(explained2, X1_cluster,
                                       title=f"Модель 2: обучена на {cluster} (вторая→первая)")
    elif plot_type == 'heatmap':
        fig_left = shap_heatmap_plot(explained1,
                                     title=f"Модель 1: обучена на {cluster} (первая→вторая)")
        fig_right = shap_heatmap_plot(explained2,
                                      title=f"Модель 2: обучена на {cluster} (вторая→первая)")
    else:
        return None, None, f"❌ Неизвестный тип графика: {plot_type}"

    return fig_left, fig_right, f"✅ SHAP графики построены для кластера {cluster}"


def shap_summary_plot(explained, data, plot_type='dot', title=None, figsize=(10, 6)):
    """Строит summary plot"""
    plt.figure(figsize=figsize)

    shap.summary_plot(
        explained.values,
        data,
        plot_type=plot_type,
        feature_names=data.columns.tolist(),
        max_display=data.shape[1],
        show=False
    )

    if title:
        plt.title(title, fontsize=12)

    plt.tight_layout()
    return plt.gcf()


def shap_decision_plot(explained, data, title=None, figsize=(12, 8)):
    """Строит decision plot"""
    plt.figure(figsize=figsize)

    shap.decision_plot(
        explained.base_values,
        explained.values,
        data.values,
        feature_names=data.columns.tolist(),
        show=False
    )

    if title:
        plt.title(title, fontsize=12)

    plt.tight_layout()
    return plt.gcf()


def shap_heatmap_plot(explained, title=None, figsize=(12, 8)):
    """Строит heatmap"""
    plt.figure(figsize=figsize)
    shap.plots.heatmap(explained, show=False)

    if title:
        plt.title(title, fontsize=12)

    plt.tight_layout()
    return plt.gcf()


def create_scatter_plot(y_true, y_pred, title):
    """Строит scatterplot реальных vs предсказанных значений"""
    plt.figure(figsize=(10, 6))

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', edgecolors='white', s=50)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')

    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_histogram_plot(y_true, y_pred, title):
    """Строит гистограммы распределений реальных и предсказанных значений"""
    plt.figure(figsize=(10, 6))

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    plt.hist(y_true, bins=30, alpha=0.5, color='blue', label='Реальные значения', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, color='red', label='Предсказанные значения', edgecolor='black')

    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_visualizations_for_cluster(data_dict, cluster, plot_type, summary_plot_type='dot'):
    """Основная функция для создания всех типов визуализаций"""
    if data_dict is None:
        return None, None, "❌ Сначала выполните разбиение данных!"

    # Приводим cluster к строке для поиска в кэше
    cluster_str = str(cluster)

    if cluster_str not in trained_models:
        return None, None, f"❌ Модели для кластера {cluster_str} не найдены в кэше! Доступные кластеры: {list(trained_models.keys())}"

    model_data = trained_models[cluster_str]
    X1 = data_dict.get('X1')
    X2 = data_dict.get('X2')
    y1 = data_dict.get('y1')
    y2 = data_dict.get('y2')

    if X1 is None or X2 is None:
        return None, None, "❌ Данные не найдены!"

    cluster_col = 'cluster'
    idx1 = X1[X1[cluster_col].astype(str) == str(cluster)].index
    idx2 = X2[X2[cluster_col].astype(str) == str(cluster)].index

    X1_cluster = X1.loc[idx1].drop(columns=[cluster_col])
    X2_cluster = X2.loc[idx2].drop(columns=[cluster_col])

    if isinstance(y1, pd.DataFrame):
        y1_cluster = y1.loc[idx1].iloc[:, 0]
    else:
        y1_cluster = y1.loc[idx1]

    if isinstance(y2, pd.DataFrame):
        y2_cluster = y2.loc[idx2].iloc[:, 0]
    else:
        y2_cluster = y2.loc[idx2]

    model1 = model_data.get('model1')
    model2 = model_data.get('model2')

    if model1 is None or model2 is None:
        return None, None, f"❌ Одна из моделей для кластера {cluster_str} не обучена!"

    fig_left = None
    fig_right = None

    if plot_type in ['scatter', 'histogram']:
        # Берём предсказания из кэша
        if cluster_str in predictions_cache:
            preds1 = predictions_cache[cluster_str]['preds1']
            preds2 = predictions_cache[cluster_str]['preds2']
        else:
            # Если почему-то нет в кэше - вычисляем и сохраняем
            preds1 = model1.predict(X2_cluster)
            preds2 = model2.predict(X1_cluster)
            predictions_cache[cluster_str] = {'preds1': preds1, 'preds2': preds2}

        if plot_type == 'scatter':
            fig_left = create_scatter_plot(y2_cluster, preds1,
                                           title=f"Модель 1: обучена на {cluster_str} (первая→вторая)")
            fig_right = create_scatter_plot(y1_cluster, preds2,
                                            title=f"Модель 2: обучена на {cluster_str} (вторая→первая)")
        else:
            fig_left = create_histogram_plot(y2_cluster, preds1,
                                             title=f"Модель 1: обучена на {cluster_str} (первая→вторая)")
            fig_right = create_histogram_plot(y1_cluster, preds2,
                                              title=f"Модель 2: обучена на {cluster_str} (вторая→первая)")

    elif plot_type in ['shap_summary', 'shap_decision', 'shap_heatmap']:
        shap_type = {
            'shap_summary': 'summary',
            'shap_decision': 'decision',
            'shap_heatmap': 'heatmap'
        }[plot_type]

        fig_left, fig_right, msg = create_shap_plots_for_cluster(
            data_dict, cluster_str, shap_type, summary_plot_type
        )
        return fig_left, fig_right, msg

    return fig_left, fig_right, f"✅ Графики ({plot_type}) построены для кластера {cluster_str}"


# ========== Функции для обновления выпадающих списков ==========

def update_column_choices(file):
    if file is None:
        return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

    df = pd.read_csv(file.name, index_col=0)
    columns = df.columns.tolist()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'cluster' in df.columns and 'cluster' not in categorical_cols:
        categorical_cols.append('cluster')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return (
        gr.update(choices=columns, value=columns[0] if columns else None),
        gr.update(choices=categorical_cols, value=categorical_cols[0] if categorical_cols else None),
        gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None)
    )


def update_cluster_choices(X1):
    """Обновляет выпадающий список кластеров"""
    clusters = get_available_clusters(X1)
    if not clusters:
        return gr.update(choices=[], value=None, interactive=False)
    return gr.update(choices=clusters, value=clusters[0] if clusters else None, interactive=True)


def update_logs(logs):
    return logs


def update_viz_summary_visibility(plot_type_val):
    return gr.update(visible=(plot_type_val == "shap_summary"))


# ========== Функция для экспорта предсказаний ==========

def export_predictions_for_cluster(data_dict, cluster, filename, predictions_column_name='2021_all'):
    """
    Экспортирует данные кластера с предсказаниями и остатками
    data_dict: словарь с данными (X1, X2, y1, y2)
    cluster: имя кластера
    filename: имя файла для экспорта
    predictions_column_name: имя колонки с предсказаниями (по умолчанию '2021_all')
    """
    if data_dict is None:
        return None, "❌ Сначала выполните разбиение данных!"

    cluster_str = str(cluster)

    if cluster_str not in trained_models:
        return None, f"❌ Модели для кластера {cluster_str} не найдены в кэше! Сначала обучите модели."

    model_data = trained_models[cluster_str]
    X1 = data_dict.get('X1')
    X2 = data_dict.get('X2')
    y1 = data_dict.get('y1')
    y2 = data_dict.get('y2')

    if X1 is None or X2 is None:
        return None, "❌ Данные не найдены!"

    # Очищаем имя файла от недопустимых символов
    if not filename:
        filename = f"cluster"
    else:
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
        if not filename:
            filename = f"cluster"

    # Очищаем имя колонки с предсказаниями
    pred_col_name = re.sub(r'[<>:"/\\|?*]', '_', predictions_column_name.strip())
    if not pred_col_name:
        pred_col_name = '2021_all'

    cluster_col = 'cluster'
    idx1 = X1[X1[cluster_col].astype(str) == cluster_str].index
    idx2 = X2[X2[cluster_col].astype(str) == cluster_str].index

    # Получаем модели
    model1 = model_data.get('model1')
    model2 = model_data.get('model2')

    if model1 is None or model2 is None:
        return None, f"❌ Одна из моделей для кластера {cluster_str} не обучена!"

    # Получаем предсказания (из кэша или вычисляем)
    if cluster_str in predictions_cache:
        preds1 = predictions_cache[cluster_str]['preds1']
        preds2 = predictions_cache[cluster_str]['preds2']
    else:
        X1_cluster = X1.loc[idx1].drop(columns=[cluster_col])
        X2_cluster = X2.loc[idx2].drop(columns=[cluster_col])
        preds1 = model1.predict(X2_cluster)
        preds2 = model2.predict(X1_cluster)
        predictions_cache[cluster_str] = {'preds1': preds1, 'preds2': preds2}

    # Получаем целевые переменные
    if isinstance(y1, pd.DataFrame):
        y1_cluster = y1.loc[idx1].iloc[:, 0]
    else:
        y1_cluster = y1.loc[idx1]

    if isinstance(y2, pd.DataFrame):
        y2_cluster = y2.loc[idx2].iloc[:, 0]
    else:
        y2_cluster = y2.loc[idx2]

    # Преобразуем в одномерные массивы
    y1_values = np.array(y1_cluster).ravel()
    y2_values = np.array(y2_cluster).ravel()
    preds1 = np.array(preds1).ravel()
    preds2 = np.array(preds2).ravel()

    # Создаём датафреймы для каждой модели
    # Модель 1: обучалась на X1, предсказывает для X2
    df_model1 = X2.loc[idx2].copy()
    df_model1[y2.name] = y2_values
    df_model1[pred_col_name] = preds1
    df_model1['resid'] = y2_values - preds1

    # Модель 2: обучалась на X2, предсказывает для X1
    df_model2 = X1.loc[idx1].copy()
    df_model2[y1.name] = y1_values
    df_model2[pred_col_name] = preds2
    df_model2['resid'] = y1_values - preds2

    # Объединяем оба датафрейма
    result_df = pd.concat([df_model1, df_model2])

    # Добавляем расширение .csv, если его нет
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Создаём файл во временной директории
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    # Сохраняем с включённым индексом
    result_df.to_csv(file_path, index=True, index_label=X1.index.name, encoding='utf-8-sig')

    # Статистика экспорта
    stats = f"✅ Экспортировано:\n"
    stats += f"   - Кластер: {cluster_str}\n"
    stats += f"   - Модель 1 (первая→вторая): {len(df_model1)} строк\n"
    stats += f"   - Модель 2 (вторая→первая): {len(df_model2)} строк\n"
    stats += f"   - Всего: {len(result_df)} строк\n"
    stats += f"   - Колонка предсказаний: '{pred_col_name}'\n"
    stats += f"   - Файл: {filename}"

    return file_path, stats


# ========== Создание интерфейса ==========

with gr.Blocks(title="Стратифицированное разбиение и обучение моделей CatBoost", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Инструмент для стратифицированного разбиения датасета и обучения моделей CatBoost")
    gr.Markdown(
        "> 💡 **Инструкция:** Загрузите CSV файл → Настройте параметры → Нажмите 'Разбить' → Выберите кластер → Настройте гиперпараметры → Обучите модели → Постройте графики")

    with gr.Tabs():
        # ===== Вкладка 1: Разбиение данных =====
        with gr.TabItem("📁 Шаг 1: Разбиение данных"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Загрузка данных")
                    file_input = gr.File(label="CSV файл", file_types=[".csv"])

                    gr.Markdown("### ⚙️ Параметры разбиения")

                    group_column = gr.Dropdown(
                        choices=[],
                        label="Колонка для группировки",
                        info="Данные будут разбиты на поддатасеты по уникальным значениям этой колонки"
                    )

                    stratify_column = gr.Dropdown(
                        choices=[],
                        label="Колонка для стратификации",
                        info="Распределение этой колонки будет сохранено в выборках"
                    )

                    second_size = gr.Slider(
                        0.1, 0.5, 0.5, 0.05,
                        label="Доля второй выборки",
                        info="Рекомендуется 0.2-0.3 для небольших датасетов"
                    )

                    target_column = gr.Dropdown(
                        choices=[],
                        label="Целевая переменная",
                        info="Должна быть числовой"
                    )

                    random_state = gr.Number(value=42, label="Random state", precision=0)

                    split_btn = gr.Button("🔪 Выполнить разбиение", variant="primary", size="lg")

                    gr.Markdown("### 📊 Визуализация распределений")
                    plot_btn = gr.Button("📈 Построить графики", variant="secondary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 📈 Результаты")
                    stats_output = gr.Textbox(label="Статистика", lines=20)
                    status_output = gr.Textbox(label="Статус", lines=2, interactive=False)

                    gr.Markdown("### 📊 Информация о выборках")
                    with gr.Row():
                        first_info = gr.Textbox(label="Первая выборка", lines=1, interactive=False)
                        second_info = gr.Textbox(label="Вторая выборка", lines=1, interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Предпросмотр первой выборки (X1)")
                    first_preview = gr.Dataframe(label="X1 (первые 5 строк)", interactive=False)
                with gr.Column():
                    gr.Markdown("### 🎯 Целевая переменная (y1)")
                    y1_preview = gr.Dataframe(label="y1 (первые 5 значений)", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Предпросмотр второй выборки (X2)")
                    second_preview = gr.Dataframe(label="X2 (первые 5 строк)", interactive=False)
                with gr.Column():
                    gr.Markdown("### 🎯 Целевая переменная (y2)")
                    y2_preview = gr.Dataframe(label="y2 (первые 5 значений)", interactive=False)

            gr.Markdown("---")
            gr.Markdown("## 📊 Визуализация распределений признаков")

            gr.Markdown("### 🥧 Категориальные признаки (сравнение Первая vs Вторая)")
            categorical_plot = gr.Plot(label="Категориальные признаки")

            gr.Markdown("### 📊 Числовые признаки (сравнение Первая vs Вторая)")
            numerical_plot = gr.Plot(label="Числовые признаки")

            plot_status = gr.Textbox(label="Статус графиков", lines=1, interactive=False)

        # ===== Вкладка 2: Обучение моделей =====
        with gr.TabItem("🤖 Шаг 2: Обучение моделей"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Выбор кластера")
                    cluster_selector = gr.Dropdown(
                        choices=[],
                        label="Выберите кластер для обучения",
                        interactive=True,
                        info="Будут обучены две модели: первая→вторая и вторая→первая"
                    )
                    refresh_clusters_btn = gr.Button("🔄 Обновить список кластеров", variant="secondary", size="sm")

                    gr.Markdown("### ⚙️ Параметры оптимизации")
                    n_trials = gr.Slider(1, 200, 50, 10, label="Количество попыток (n_trials)")
                    cv = gr.Slider(2, 10, 5, 1, label="Количество фолдов (cv)")

                    gr.Markdown("### 🎯 Метрика оптимизации")
                    direction = gr.Radio(
                        choices=["maximize", "minimize"],
                        label="Направление оптимизации",
                        value="maximize"
                    )
                    scoring = gr.Dropdown(
                        choices=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
                        label="Метрика (scoring)",
                        value="neg_mean_squared_error"
                    )

                    gr.Markdown("### 📊 Гиперпараметры для перебора")

                    gr.Markdown("#### n_estimators (количество деревьев)")
                    with gr.Row():
                        n_estimators_min = gr.Number(value=100, label="Min", precision=0)
                        n_estimators_max = gr.Number(value=1000, label="Max", precision=0)
                    n_estimators_log = gr.Checkbox(label="Логарифмический масштаб", value=True)

                    gr.Markdown("#### depth (глубина деревьев)")
                    with gr.Row():
                        depth_min = gr.Number(value=2, label="Min", precision=0)
                        depth_max = gr.Number(value=12, label="Max", precision=0)

                    gr.Markdown("#### l2_leaf_reg (L2 регуляризация)")
                    with gr.Row():
                        l2_leaf_reg_min = gr.Number(value=0.01, label="Min", precision=2, step=0.01)
                        l2_leaf_reg_max = gr.Number(value=1.0, label="Max", precision=2, step=0.01)
                    l2_leaf_reg_log = gr.Checkbox(label="Логарифмический масштаб", value=True)

                    gr.Markdown("#### learning_rate (скорость обучения)")
                    with gr.Row():
                        learning_rate_min = gr.Number(value=0.01, label="Min", precision=2, step=0.01)
                        learning_rate_max = gr.Number(value=1.0, label="Max", precision=2, step=0.01)
                    learning_rate_log = gr.Checkbox(label="Логарифмический масштаб", value=True)

                    gr.Markdown("#### min_data_in_leaf (мин. данных в листе)")
                    with gr.Row():
                        min_data_in_leaf_min = gr.Number(value=1, label="Min", precision=0)
                        min_data_in_leaf_max = gr.Number(value=50, label="Max", precision=0)

                    train_btn = gr.Button("🚀 Обучить модели для выбранного кластера", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 📈 Результаты обучения")
                    train_result = gr.Textbox(label="Детали обучения", lines=20)
                    train_status = gr.Textbox(label="Статус", lines=2, interactive=False)
                    train_logs = gr.Textbox(label="Логи Optuna (пошагово)", lines=15, interactive=False)

                    gr.Markdown("### 📊 Информация о кластере")
                    with gr.Row():
                        cluster_first_size = gr.Number(label="Размер первой выборки", interactive=False)
                        cluster_second_size = gr.Number(label="Размер второй выборки", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Прогнозы (Модель 1: первая→вторая)")
                    preds1_preview = gr.Dataframe(label="Предсказания для второй выборки", interactive=False)
                with gr.Column():
                    gr.Markdown("### 🔍 Прогнозы (Модель 2: вторая→первая)")
                    preds2_preview = gr.Dataframe(label="Предсказания для первой выборки", interactive=False)

        # ===== Вкладка 3: Визуализация =====
        with gr.TabItem("📈 Шаг 3: Визуализация"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Выбор кластера")
                    viz_cluster_selector = gr.Dropdown(
                        choices=[],
                        label="Выберите кластер",
                        interactive=True,
                        info="Будут показаны графики для обеих моделей кластера"
                    )
                    refresh_viz_clusters_btn = gr.Button("🔄 Обновить список кластеров", variant="secondary", size="sm")

                    gr.Markdown("### 📊 Тип визуализации")
                    plot_type = gr.Radio(
                        choices=["scatter", "histogram", "shap_summary", "shap_decision", "shap_heatmap"],
                        label="Выберите тип графика",
                        value="scatter"
                    )

                    with gr.Group(visible=False) as viz_shap_summary_group:
                        summary_plot_type = gr.Radio(
                            choices=["dot", "bar", "violin"],
                            label="Тип summary plot",
                            value="dot"
                        )

                    viz_btn = gr.Button("📈 Построить графики", variant="primary", size="lg")

                with gr.Column(scale=1):
                    viz_status = gr.Textbox(label="Статус", lines=3, interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Модель 1 (обучена на первой, предсказывает для второй)")
                    plot_left = gr.Plot(label="Модель 1")
                with gr.Column():
                    gr.Markdown("### 📊 Модель 2 (обучена на второй, предсказывает для первой)")
                    plot_right = gr.Plot(label="Модель 2")

        # ===== Вкладка 4: Управление кэшем =====
        with gr.TabItem("🗂️ Шаг 4: Кэш моделей"):
            with gr.Row():
                with gr.Column():
                    show_btn = gr.Button("👁️ Показать все модели", variant="secondary")
                    cache_info = gr.Textbox(label="Информация о кэше", lines=15, interactive=False)
                    clear_btn = gr.Button("🗑️ Очистить кэш", variant="stop")
                    clear_result = gr.Textbox(label="Результат очистки", lines=3, interactive=False)

        # ===== Вкладка 5: Экспорт предсказаний =====
        with gr.TabItem("💾 Шаг 5: Экспорт предсказаний"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Выбор кластера")
                    export_cluster_selector = gr.Dropdown(
                        choices=[],
                        label="Выберите кластер для экспорта",
                        interactive=True,
                        info="Будут экспортированы предсказания обеих моделей"
                    )
                    refresh_export_clusters_btn = gr.Button("🔄 Обновить список кластеров", variant="secondary",
                                                            size="sm")

                    gr.Markdown("### ⚙️ Параметры экспорта")
                    export_filename = gr.Textbox(
                        label="Имя файла",
                        placeholder="cluster (без расширения или с .csv)",
                        value="cluster"
                    )

                    predictions_column_name = gr.Textbox(
                        label="Название колонки с предсказаниями",
                        placeholder="2021_all",
                        value="2021_all",
                        info="Эта колонка будет добавлена в датасет с предсказаниями модели"
                    )

                    export_btn = gr.Button("📥 Скачать предсказания (CSV)", variant="primary", size="lg")

                with gr.Column(scale=1):
                    export_file = gr.File(label="Скачать CSV", visible=True)
                    export_status = gr.Textbox(label="Статус экспорта", lines=8, interactive=False)

    # Скрытые состояния
    X1_state = gr.State()
    y1_state = gr.State()
    X2_state = gr.State()
    y2_state = gr.State()
    df1_state = gr.State()
    df2_state = gr.State()
    data_dict_state = gr.State()

    # Обработчики для вкладки 1
    file_input.change(
        update_column_choices,
        inputs=[file_input],
        outputs=[group_column, stratify_column, target_column]
    )

    split_btn.click(
        split_data,
        inputs=[file_input, group_column, stratify_column, second_size, target_column, random_state],
        outputs=[
            X1_state, y1_state, X2_state, y2_state,
            df1_state, df2_state, data_dict_state,
            stats_output, status_output,
            first_info, second_info
        ]
    ).then(
        lambda X1, y1, X2, y2: (
            X1.head() if X1 is not None else None,
            y1.head() if y1 is not None else None,
            X2.head() if X2 is not None else None,
            y2.head() if y2 is not None else None
        ),
        inputs=[X1_state, y1_state, X2_state, y2_state],
        outputs=[first_preview, y1_preview, second_preview, y2_preview]
    ).then(
        lambda X1: update_cluster_choices(X1),
        inputs=[X1_state],
        outputs=[cluster_selector]
    ).then(
        lambda X1: update_cluster_choices(X1),
        inputs=[X1_state],
        outputs=[viz_cluster_selector]
    )

    plot_btn.click(
        create_visualizations,
        inputs=[df1_state, df2_state],
        outputs=[categorical_plot, numerical_plot, plot_status]
    )

    # Обработчики для вкладки 2
    refresh_clusters_btn.click(
        lambda X1: update_cluster_choices(X1),
        inputs=[X1_state],
        outputs=[cluster_selector]
    )

    train_btn.click(
        train_models_for_cluster,
        inputs=[
            X1_state, y1_state, X2_state, y2_state,
            cluster_selector,
            n_trials, cv, direction, scoring,
            n_estimators_min, n_estimators_max, n_estimators_log,
            depth_min, depth_max,
            l2_leaf_reg_min, l2_leaf_reg_max, l2_leaf_reg_log,
            learning_rate_min, learning_rate_max, learning_rate_log,
            min_data_in_leaf_min, min_data_in_leaf_max,
            random_state
        ],
        outputs=[train_result, preds1_preview, preds2_preview, train_status, cluster_first_size, cluster_second_size,
                 train_logs]
    )

    # Обработчики для вкладки 3
    refresh_viz_clusters_btn.click(
        lambda X: update_cluster_choices(X),
        inputs=[X1_state],
        outputs=[viz_cluster_selector]
    )

    plot_type.change(update_viz_summary_visibility, plot_type, viz_shap_summary_group)

    viz_btn.click(
        create_visualizations_for_cluster,
        inputs=[data_dict_state, viz_cluster_selector, plot_type, summary_plot_type],
        outputs=[plot_left, plot_right, viz_status]
    )

    # Обработчики для вкладки 4
    show_btn.click(show_cached_models, outputs=[cache_info])
    clear_btn.click(clear_cache, outputs=[clear_result])

    # Для вкладки экспорта (добавить после других обработчиков)
    refresh_export_clusters_btn.click(
        lambda X: update_cluster_choices(X),
        inputs=[X1_state],
        outputs=[export_cluster_selector]
    )

    export_btn.click(
        export_predictions_for_cluster,
        inputs=[data_dict_state, export_cluster_selector, export_filename, predictions_column_name],
        outputs=[export_file, export_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)