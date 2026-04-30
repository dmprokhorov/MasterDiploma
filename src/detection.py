# detection.py - Детектор аномалий с оптимизацией гиперпараметров и SHAP визуализациями
import gradio as gr

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

import optuna

import shap

import torch

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
import tempfile
from io import StringIO
import math
import re
import os

# ========== Глобальные переменные ==========
trained_models = {}
shap_explanations = {}  # Словарь для хранения SHAP объяснений: {model_key: explained}
log_capture = StringIO()


# ========== Метрика качества ==========
def anomaly_isolation_ratio_score(X, labels):
    """Метрика качества для аномалий"""
    normal = X.iloc[np.where(labels != -1)[0]]
    anomalies = X.iloc[np.where(labels == -1)[0]]

    if len(anomalies) == 0 or len(normal) == 0:
        return 0.0

    normal_center = np.mean(normal, axis=0)
    dist_to_center = np.mean(np.linalg.norm(anomalies - normal_center, axis=1))
    normal_compactness = np.mean(np.linalg.norm(normal - normal_center, axis=1))

    if normal_compactness == 0:
        return 0.0

    return dist_to_center / normal_compactness


# ========== Целевая функция для Optuna ==========
def objective(
        trial,
        X,
        cls,
        borders=(0.03, 0.05),
        penalty=0,
        make_negative=False,
        **params
):
    """Целевая функция для Optuna"""
    trial_params = {}
    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            if len(value) >= 3 and isinstance(value[2], bool):
                log = value[2]
            else:
                log = False

            if value[1] == float:
                trial_params[key] = trial.suggest_float(
                    key,
                    value[0][0],
                    value[0][1],
                    log=log
                )
            elif value[1] == int:
                trial_params[key] = trial.suggest_int(
                    key,
                    value[0][0],
                    value[0][1],
                    log=log
                )
            else:
                trial_params[key] = trial.suggest_categorical(key, value[0])
        else:
            trial_params[key] = value

    model = cls(**trial_params)
    labels = model.fit_predict(X)

    if make_negative:
        labels *= -1

    borders_int = tuple(map(lambda b: int(b * X.shape[0]), borders))
    n_anomalies = np.sum(labels == -1)

    if not (borders_int[0] <= n_anomalies <= borders_int[1]):
        return penalty

    return anomaly_isolation_ratio_score(X, labels)


def capture_optuna_logs(study, trial):
    """Callback для захвата логов Optuna"""
    log_capture.write(f"Попытка {trial.number}: значение = {trial.value:.4f}\n")
    log_capture.write(f"  Параметры: {trial.params}\n")
    log_capture.write("-" * 50 + "\n")


# ========== SHAP функции для визуализации ==========
def get_shap_explained(attribute, data, indices, type_explainer='Tree', convert=False, **explainer_params):
    """Возвращает SHAP объяснения для аномалий"""
    feature_names = list(data.columns)
    outliers = data.iloc[indices].values

    if convert:
        data = torch.FloatTensor(data.values)
        outliers = torch.FloatTensor(outliers)

    explainer = getattr(shap, f'{type_explainer}Explainer')(
        attribute,
        data,
        **explainer_params
    )
    explained = explainer(outliers)
    explained.feature_names = feature_names
    return explained


def shap_summary_plot(explained, data, indices, plot_type='dot', plot_size=(10, 6)):
    """Строит summary plot для SHAP"""
    plt.figure(figsize=plot_size)
    shap.summary_plot(
        explained.values,
        data.iloc[indices],
        plot_type=plot_type,
        feature_names=explained.feature_names,
        max_display=data.shape[1],
        show=False
    )
    plt.tight_layout()
    return plt.gcf()


def shap_decision_plot(explained, data, indices, model=None, plot_size=(12, 8)):
    """Строит decision plot для SHAP"""
    plt.figure(figsize=plot_size)

    if model is not None and (not hasattr(explained, 'base_values') or explained.base_values is None):
        with torch.no_grad():
            output = model(torch.FloatTensor(data.values)).numpy()
            explained.base_values = np.square(output - data.values).mean(axis=-1)

    base_value = explained.base_values.mean() if hasattr(explained,
                                                         'base_values') and explained.base_values is not None else 0
    shap.decision_plot(
        base_value,
        explained.values,
        data.iloc[indices].values,
        feature_names=data.columns.tolist(),
        show=False
    )
    plt.tight_layout()
    return plt.gcf()


def shap_heatmap_plot(explained, plot_size=(12, 8)):
    """Строит heatmap для SHAP"""
    plt.figure(figsize=plot_size)

    if explained.values.ndim > 2:
        explained.values = explained.values.mean(axis=-1)

    shap.plots.heatmap(
        explained,
        show=False
    )
    plt.tight_layout()
    return plt.gcf()


# ========== Функция для отображения списка моделей ==========
def get_trained_models_list():
    """Возвращает список всех обученных моделей для отображения"""
    if not trained_models:
        return []

    models_list = []
    for key, data in trained_models.items():
        model_name = f'{key}: {"_".join([f"{k}={v}" for k, v in data["params"].items()])}'
        n_anomalies = sum(data['predictions'] == -1)
        models_list.append(f"{model_name} | аномалий: {n_anomalies}")

    return models_list


def refresh_predict_models():
    """Обновляет список моделей для вкладки предсказаний"""
    models = get_trained_models_list()
    if not models:
        return gr.update(choices=[], value=None,
                         interactive=False), "📭 Нет обученных моделей. Сначала запустите оптимизацию!"
    else:
        return gr.update(choices=models, value=models[0] if models else None,
                         interactive=True), f"✅ Доступно моделей: {len(models)}"


def refresh_plot_models():
    """Обновляет список моделей для вкладки визуализации"""
    models = get_trained_models_list()
    if not models:
        return gr.update(choices=[], value=None,
                         interactive=False), "📭 Нет обученных моделей. Сначала запустите оптимизацию!"
    else:
        return gr.update(choices=models, value=models[0] if models else None,
                         interactive=True), f"✅ Доступно моделей: {len(models)}"


def refresh_ensemble_models():
    """Обновляет список моделей для вкладки ансамбля"""
    models = get_trained_models_list()
    if not models:
        return gr.update(choices=[], value=[]), "📭 Нет обученных моделей. Сначала запустите оптимизацию!"
    else:
        return gr.update(choices=models, value=[]), f"✅ Доступно моделей: {len(models)}. Выберите модели для ансамбля."


def show_cached_models():
    """Показывает информацию о всех обученных моделях в кэше"""
    if not trained_models:
        return "📭 Кэш пуст. Сначала запустите оптимизацию."

    result = "🗂️ Кэшированные моде:\n\n"
    for i, (key, data) in enumerate(trained_models.items(), 1):
        result += f"{i}. {key}\n"
        result += f"   Параметры: {data['params']}\n"
        result += f"   Колонки: {data['columns'][:3]}..."
        n_anomalies = sum(data['predictions'] == -1)
        result += f"   Аномалий: {n_anomalies}\n\n"

    result += f"\n📊 Всего моделей в кэше: {len(trained_models)}"
    return result


def clear_cache():
    """Очищает кэш моделей и SHAP объяснений"""
    trained_models.clear()
    shap_explanations.clear()
    return "🗑️ Кэш моделей и SHAP объяснений очищен!"


# ========== Шаг 1: Загрузка данных ==========
def load_data(file):
    """Загружает данные, нормализует и очищает кэш"""
    if file is None:
        return None, "❌ Файл не выбран", None, None, gr.update(choices=[], value=[]), "❌ Нет данных"

    df = pd.read_csv(file.name, index_col=0)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return None, "❌ Нет числовых колонок!", None, None, gr.update(choices=[], value=[]), "❌ Нет числовых колонок"

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_df.columns, index=df.index)

    available_columns = numeric_df.columns.tolist()

    data_dict = {
        'original': df,
        'normalized': normalized_df,
        'scaler': scaler,
        'all_columns': available_columns,
        'selected_columns': available_columns.copy()
    }

    info = f"✅ Загружено {df.shape[0]} строк, {df.shape[1]} колонок\n"
    info += f"📊 Числовых колонок: {len(available_columns)}\n"
    info += f"📊 Нормализация выполнена (StandardScaler)\n"
    info += f"📊 Доступные колонки: {', '.join(available_columns[:5])}"
    if len(available_columns) > 5:
        info += f" и ещё {len(available_columns) - 5}"

    trained_models.clear()
    shap_explanations.clear()
    info += "\n\n🔄 Кэш моделей и SHAP объяснений очищен"

    return data_dict, info, df.head(), normalized_df[available_columns].head(), gr.update(choices=available_columns,
                                                                                          value=available_columns), f"✅ Выбрано колонок: {len(available_columns)}"


def update_selected_columns(data_dict, columns):
    """Обновляет список выбранных колонок и показывает нормализованные данные"""
    if data_dict is None:
        return data_dict, "❌ Нет данных", None

    if not columns:
        return data_dict, "⚠️ Не выбрано ни одной колонки!", None

    data_dict['selected_columns'] = columns
    normalized_preview = data_dict['normalized'][columns].head()
    trained_models.clear()
    shap_explanations.clear()

    return data_dict, f"✅ Выбрано колонок: {len(columns)} (кэш очищен)", normalized_preview


# ========== Шаг 2: Оптимизация и обучение ==========
def run_optimization(
        data_dict,
        cls,
        n_trials,
        borders_low,
        borders_high,
        penalty,
        log_callback=None,
        **params
):
    """Запускает оптимизацию гиперпараметров и сохраняет лучшую модель"""
    if data_dict is None:
        return "❌ Сначала загрузите данные!", None, None, ""

    selected_columns = data_dict.get('selected_columns', [])
    if not selected_columns:
        return "❌ Выберите колонки для обучения!", None, None, ""

    X = data_dict['normalized'][selected_columns].copy()
    default_params = {
        key: value for key, value in params.items()
        if not isinstance(value, (tuple, list))
    }

    # Очищаем буфер логов
    log_capture.truncate(0)
    log_capture.seek(0)

    # Перенаправляем stdout для захвата вывода Optuna
    old_stdout = sys.stdout
    sys.stdout = log_capture

    try:
        study = optuna.create_study(
            direction='maximize',
            study_name=f'Оптимизация алгоритма {cls.__name__}'
        )

        def objective_wrapper(trial):
            return objective(
                trial,
                X=X,
                cls=cls,
                borders=(borders_low, borders_high),
                penalty=penalty,
                make_negative=False,
                **params
            )

        study.optimize(
            objective_wrapper,
            n_trials=int(n_trials),
            callbacks=[capture_optuna_logs],
            show_progress_bar=True
        )

        logs = log_capture.getvalue()

    finally:
        sys.stdout = old_stdout

    # Обновляем лог в интерфейсе
    if log_callback:
        log_callback(logs)

    best_params = study.best_params
    final_params = {**best_params, **default_params}

    model = cls(**final_params)
    predictions = model.fit_predict(X)

    n_anomalies = sum(predictions == -1)
    anomaly_indices = np.where(predictions == -1)[0].tolist()
    score = anomaly_isolation_ratio_score(X, predictions)

    # Для LOF создаём дополнительную модель с novelty=True
    novelty_model = None
    if cls.__name__ == "LocalOutlierFactor":
        novelty_model = LocalOutlierFactor(**final_params, novelty=True)
        novelty_model.fit(X)

    # Сохраняем только одну модель на класс (затираем старую)
    model_key = cls.__name__
    trained_models[model_key] = {
        'model': model,
        'novelty_model': novelty_model,
        'predictions': predictions,
        'params': final_params,
        'columns': selected_columns.copy(),
        'anomaly_indices': anomaly_indices
    }

    # Удаляем старые SHAP объяснения для этого алгоритма
    if model_key in shap_explanations:
        del shap_explanations[model_key]

    result = f"🏆 Оптимизация завершена!\n\n"
    result += f"📊 Лучшие параметры:\n"
    for key, value in best_params.items():
        result += f"   {key}: {value}\n"
    result += f"\n📈 Лучшее значение метрики: {score:.4f}\n"
    result += f"🔴 Найдено аномалий: {n_anomalies} ({n_anomalies / len(X) * 100:.1f}%)\n"
    result += f"📋 Индексы аномалий (первые 20): {anomaly_indices[:20]}\n"
    result += f"\n📊 Количество попыток: {n_trials}"

    params_text = "\n".join([f"{k}: {v}" for k, v in best_params.items()])

    return result, params_text, score, logs


# ========== Шаг 3: Предсказания ==========
def get_predictions_from_selected_model(data_dict, selected_model_display):
    """Возвращает предсказания для выбранной модели"""
    if data_dict is None:
        return "❌ Сначала загрузите данные!"

    if not selected_model_display:
        return "❌ Выберите модель из списка!"

    selected_key = None
    for key, data in trained_models.items():
        model_name = f'{key}: {"_".join([f"{k}={v}" for k, v in data["params"].items()])}'
        model_display = f"{model_name} | аномалий: {sum(data['predictions'] == -1)}"

        if model_display == selected_model_display:
            selected_key = key
            break

    if selected_key is None:
        return "❌ Модель не найдена в кэше!"

    model_data = trained_models[selected_key]
    predictions = model_data['predictions']
    params = model_data['params']
    df = data_dict['original']

    anomaly_mask = predictions == -1
    n_anomalies = sum(anomaly_mask)
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    normal_indices = np.where(predictions == 1)[0].tolist()

    result = f"🔍 Модель: {selected_key}\n"
    result += f"⚙️ Параметры: {params}\n"
    result += f"📊 Колонки: {model_data['columns'][:5]}"
    if len(model_data['columns']) > 5:
        result += f" и ещё {len(model_data['columns']) - 5}\n"
    else:
        result += "\n"
    result += f"\n📊 Результаты предсказания:\n"
    result += f"   ✅ Нормальные: {len(normal_indices)} ({len(normal_indices) / len(df) * 100:.1f}%)\n"
    result += f"   🔴 Аномалии: {n_anomalies} ({n_anomalies / len(df) * 100:.1f}%)\n"

    result += f"\n📋 Индексы аномалий (первые 30):\n"
    if anomaly_indices:
        result += f"   {anomaly_indices[:30]}"
        if len(anomaly_indices) > 30:
            result += f"\n   ... и ещё {len(anomaly_indices) - 30}"
    else:
        result += f"   Аномалий не найдено"

    return result


# ========== Шаг 4: Визуализация ==========
def get_model_data(selected_model_display):
    """Получает данные модели по отображаемому имени"""
    for key, data in trained_models.items():
        model_name = f'{key}: {"_".join([f"{k}={v}" for k, v in data["params"].items()])}'
        model_display = f"{model_name} | аномалий: {sum(data['predictions'] == -1)}"
        if model_display == selected_model_display:
            return key, data
    return None, None


def create_pca_plot(data_dict, selected_model_display):
    """Строит график PCA для выбранной модели"""
    if data_dict is None:
        return None, "❌ Сначала загрузите данные!"

    if not selected_model_display:
        return None, "❌ Выберите модель из списка!"

    model_key, model_data = get_model_data(selected_model_display)
    if model_key is None:
        return None, "❌ Модель не найдена!"

    predictions = model_data['predictions']
    model_columns = model_data['columns']
    normalized_df = data_dict['normalized'][model_columns].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if normalized_df.shape[1] >= 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(normalized_df)

        normal = data_2d[predictions == 1]
        anomalies = data_2d[predictions == -1]

        ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='Нормальные', alpha=0.6, s=50)
        ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Аномалии', marker='x', s=100, linewidths=2)

        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
    else:
        ax.plot(predictions, 'o-', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.fill_between(range(len(predictions)), -1, 1, where=(predictions == -1), color='red', alpha=0.3)
        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Предсказание (1=норма, -1=аномалия)')

    n_anomalies = sum(predictions == -1)
    ax.set_title(f'{model_key}\nКолонки: {model_columns[:3]}...\nАномалий: {n_anomalies}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, f"✅ График PCA для модели: {selected_model_display}"


def create_shap_plot(data_dict, selected_model_display, plot_type, summary_plot_type='dot'):
    """Создаёт SHAP график для выбранной модели"""
    if data_dict is None:
        return None, "❌ Сначала загрузите данные!"

    if not selected_model_display:
        return None, "❌ Выберите модель из списка!"

    model_key, model_data = get_model_data(selected_model_display)
    if model_key is None:
        return None, "❌ Модель не найдена!"

    # Определяем тип объяснителя для алгоритма
    if model_key == "OneClassSVM":
        type_explainer = 'Kernel'
    elif model_key == "IsolationForest":
        type_explainer = 'Tree'
    else:  # LocalOutlierFactor
        type_explainer = 'Kernel'

    # Получаем данные
    X = data_dict['normalized'][model_data['columns']].copy()
    anomaly_indices = model_data['anomaly_indices']

    if len(anomaly_indices) == 0:
        return None, "⚠️ Нет аномалий для объяснения!"

    # Проверяем, есть ли уже сохранённые объяснения
    if model_key in shap_explanations:
        explained = shap_explanations[model_key]
    else:
        # Выбираем модель для атрибута
        if model_key == "LocalOutlierFactor" and model_data.get('novelty_model') is not None:
            attribute_model = model_data['novelty_model']
        else:
            attribute_model = model_data['model']

        # Получаем атрибут (метод score_samples)
        if model_key == "IsolationForest":
            attribute = attribute_model
        elif hasattr(attribute_model, 'score_samples'):
            attribute = attribute_model.score_samples
        else:
            attribute = attribute_model.decision_function

        # Вычисляем SHAP объяснения
        explained = get_shap_explained(
            attribute=attribute,
            data=X,
            indices=anomaly_indices,
            type_explainer=type_explainer,
            convert=(type_explainer == 'Deep')
        )
        shap_explanations[model_key] = explained

    # Создаём график в зависимости от типа
    if plot_type == 'summary':
        fig = shap_summary_plot(explained, X, anomaly_indices, summary_plot_type)
        return fig, f"✅ SHAP summary plot ({summary_plot_type}) для модели: {selected_model_display}"
    elif plot_type == 'decision':
        fig = shap_decision_plot(explained, X, anomaly_indices, model=None)
        return fig, f"✅ SHAP decision plot для модели: {selected_model_display}"
    elif plot_type == 'heatmap':
        fig = shap_heatmap_plot(explained)
        return fig, f"✅ SHAP heatmap для модели: {selected_model_display}"
    else:
        return None, f"❌ Неизвестный тип графика: {plot_type}"


# ========== Шаг 4: Визуализация ==========
# ========== Функции для визуализации распределений аномалий ==========
def plot_numerical_distributions(
        data,
        indices,
        columns,
        figsize=(18, 10),
        palette='rocket'
):
    """Строит гистограммы для численных признаков аномалий"""
    if len(columns) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Нет числовых признаков для отображения",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    ncols = min(int(math.ceil(len(columns) ** 0.5)), 4)
    nrows = max(1, math.ceil(len(columns) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Преобразуем axes в плоский список для удобства
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        if i < len(axes):
            import seaborn as sns
            sns.histplot(
                data.iloc[indices][column],
                kde=True,
                color='red',
                alpha=0.6,
                ax=axes[i]
            )
            axes[i].set_xlabel(None)
            axes[i].set_title(column, fontsize=10)

    # Скрываем лишние подграфики
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Распределение числовых признаков среди аномалий (n={len(indices)})', fontsize=14)
    plt.tight_layout()
    return fig


def plot_categorical_distributions(
        data,
        indices,
        columns,
        figsize=(18, 10),
        palette='rocket'
):
    """Строит столбчатые диаграммы для категориальных признаков аномалий"""
    if len(columns) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Нет категориальных признаков для отображения",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    ncols = min(int(math.ceil(len(columns) ** 0.5)), 4)
    nrows = max(1, math.ceil(len(columns) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        if i < len(axes):
            import seaborn as sns
            value_counts = data.iloc[indices][column].value_counts()
            colors = sns.color_palette(palette, len(value_counts))

            bars = axes[i].bar(range(len(value_counts)), value_counts.values, color=colors, alpha=0.7)
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
            axes[i].set_title(column, fontsize=10)
            axes[i].set_ylabel('Количество')

            # Добавляем подписи на столбцах
            for bar, val in zip(bars, value_counts.values):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                             str(val), ha='center', va='bottom', fontsize=8)

    # Скрываем лишние подграфики
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Распределение категориальных признаков среди аномалий (n={len(indices)})', fontsize=14)
    plt.tight_layout()
    return fig


def plot_scores(model, data, indices, method='score_samples', figsize=(14, 8)):
    """Строит точечную диаграмму значений выбросности объектов"""
    fig, ax = plt.subplots(figsize=figsize)

    scores = pd.Series(
        getattr(model, method)(data),
        index=data.index
    )

    # Аномалии (оранжевые)
    ax.scatter(
        indices,
        scores.iloc[indices],
        color='orange',
        label='Аномалии',
        s=50,
        alpha=0.7
    )

    # Нормальные объекты (синие)
    normal_indices = np.delete(range(data.shape[0]), indices)
    ax.scatter(
        normal_indices,
        scores.iloc[normal_indices],
        color='blue',
        label='Норма',
        s=30,
        alpha=0.5
    )

    ax.set_xlabel('Индекс объекта', fontsize=12)
    ax.set_ylabel('Значение "выбросности" (score)', fontsize=12)
    ax.set_title(f'Точечная диаграмма значений "выбросности" объектов', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_score_plot(data_dict, selected_model_display):
    """Создаёт график выбросности для выбранной модели"""
    if data_dict is None:
        return None, "❌ Сначала загрузите данные!"

    if not selected_model_display:
        return None, "❌ Выберите модель из списка!"

    model_key, model_data = get_model_data(selected_model_display)
    if model_key is None:
        return None, "❌ Модель не найдена!"

    # Получаем данные
    X = data_dict['normalized'][model_data['columns']].copy()
    anomaly_indices = model_data['anomaly_indices']

    if len(anomaly_indices) == 0:
        return None, "⚠️ Нет аномалий для отображения!"

    # Выбираем модель для получения scores
    model = model_data['model']
    if model_key == 'LocalOutlierFactor':
        model = model_data['novelty_model']
    method = 'score_samples' if hasattr(model, 'score_samples') else 'decision_function'

    fig = plot_scores(model, X, anomaly_indices, method=method)
    return fig, f"✅ График выбросности для модели: {selected_model_display}"


# ========== Шаг 5: Ансамбль аномалий ==========
def ensemble_prediction(data_dict, selected_models_display, threshold, show_numerical, show_categorical):
    """Выполняет ансамблевое предсказание и сохраняет результаты в state"""
    if data_dict is None:
        return "❌ Сначала загрузите данные!", "", None, None, None, "❌ Нет данных", None

    if not selected_models_display:
        return "❌ Выберите хотя бы одну модель!", "", None, None, None, "❌ Модели не выбраны", None

    # Находим модели по отображаемым именам
    selected_keys = []
    for display_name in selected_models_display:
        for key, data in trained_models.items():
            model_name = f'{key}: {"_".join([f"{k}={v}" for k, v in data["params"].items()])}'
            model_display = f"{model_name} | аномалий: {sum(data['predictions'] == -1)}"
            if model_display == display_name:
                selected_keys.append(key)
                break

    if not selected_keys:
        return "❌ Выбранные модели не найдены в кэше!", "", None, None, None, "❌ Ошибка", None

    # Собираем предсказания всех выбранных моделей
    all_predictions = []
    model_names = []
    for key in selected_keys:
        model_data = trained_models[key]
        all_predictions.append(model_data['predictions'])
        model_names.append(key)

    all_predictions = np.array(all_predictions)
    n_models = len(all_predictions)
    n_samples = all_predictions.shape[1]

    anomaly_votes = np.sum(all_predictions == -1, axis=0)
    anomaly_ratio = anomaly_votes / n_models

    ensemble_labels = (anomaly_ratio >= threshold).astype(int)
    ensemble_labels = np.where(ensemble_labels == 1, -1, 1)

    n_anomalies = sum(ensemble_labels == -1)
    anomaly_indices = np.where(ensemble_labels == -1)[0].tolist()
    anomaly_mask = ensemble_labels == -1

    # Сохраняем результаты для экспорта
    ensemble_results = {
        'anomaly_mask': anomaly_mask,
        'anomaly_indices': anomaly_indices,
        'selected_keys': selected_keys,
        'threshold': threshold,
        'n_anomalies': n_anomalies,
        'model_names': model_names,
        'n_models': n_models,
        'n_samples': n_samples,
        'anomaly_votes': anomaly_votes,
        'anomaly_ratio': anomaly_ratio
    }

    # Результат
    result = f"🎯 Ансамблевое предсказание\n"
    result += f"{'=' * 50}\n\n"
    result += f"📊 Участвующие модели ({n_models}):\n"
    for name in model_names:
        result += f"   - {name}\n"
    result += f"\n⚙️ Порог голосования (p): {threshold}\n"
    result += f"\n📊 Результаты:\n"
    result += f"   ✅ Нормальные: {n_samples - n_anomalies} ({(n_samples - n_anomalies) / n_samples * 100:.1f}%)\n"
    result += f"   🔴 Аномалии: {n_anomalies} ({n_anomalies / n_samples * 100:.1f}%)\n"

    result += f"\n📋 Индексы аномалий (первые 30):\n"
    if anomaly_indices:
        result += f"   {anomaly_indices[:30]}"
        if len(anomaly_indices) > 30:
            result += f"\n   ... и ещё {len(anomaly_indices) - 30}"
    else:
        result += f"   Аномалий не найдено"

    # Статистика голосования
    vote_distribution = np.bincount(anomaly_votes, minlength=n_models + 1)
    stats = f"📊 Статистика голосования:\n"
    stats += f"{'=' * 30}\n"
    for i in range(n_models + 1):
        if vote_distribution[i] > 0:
            stats += f"   {i}/{n_models} алгоритмов: {vote_distribution[i]} образцов ({vote_distribution[i] / n_samples * 100:.1f}%)\n"

    # Визуализация PCA
    first_model_data = trained_models[selected_keys[0]]
    model_columns = first_model_data['columns']
    normalized_df = data_dict['normalized'][model_columns].copy()

    fig_pca, ax = plt.subplots(figsize=(10, 6))

    if normalized_df.shape[1] >= 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(normalized_df)

        normal = data_2d[ensemble_labels == 1]
        anomalies = data_2d[ensemble_labels == -1]

        ax.scatter(normal[:, 0], normal[:, 1], c='blue', label=f'Нормальные ({n_samples - n_anomalies})',
                   alpha=0.6, s=50)
        ax.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label=f'Аномалии ({n_anomalies})', marker='x',
                   s=100, linewidths=2, alpha=0.8)

        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
        ax.set_title(f'Ансамбль ({n_models} моделей), порог p={threshold}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.plot(ensemble_labels, 'o-', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.fill_between(range(len(ensemble_labels)), -1, 1, where=(ensemble_labels == -1), color='red', alpha=0.3)
        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Предсказание (1=норма, -1=аномалия)')
        ax.set_title(f'Ансамбль ({n_models} моделей), порог p={threshold}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Визуализация распределений (если есть аномалии)
    fig_numerical = None
    fig_categorical = None

    original_df = data_dict['original']

    if n_anomalies > 0:
        if show_numerical:
            numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                fig_numerical = plot_numerical_distributions(
                    original_df, anomaly_indices, numerical_cols
                )
            else:
                fig_numerical = plot_numerical_distributions(original_df, anomaly_indices, [])

        if show_categorical:
            categorical_cols = original_df.select_dtypes(include=['str', 'object']).columns.tolist()
            if categorical_cols:
                fig_categorical = plot_categorical_distributions(
                    original_df, anomaly_indices, categorical_cols
                )
            else:
                fig_categorical = plot_categorical_distributions(original_df, anomaly_indices, [])
    else:
        if show_numerical or show_categorical:
            empty_fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Нет аномалий для визуализации распределений",
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            if show_numerical:
                fig_numerical = empty_fig
            if show_categorical:
                fig_categorical = empty_fig

    status = f"✅ График PCA построен для {n_models} моделей, порог {threshold}"
    if n_anomalies > 0:
        status += f", найдено {n_anomalies} аномалий"
    else:
        status += f", аномалий не найдено"

    return result, stats, fig_pca, fig_numerical, fig_categorical, status, ensemble_results


# ========== Функция для экспорта аномалий ==========
def export_anomalies_to_csv(data_dict, ensemble_results, filename):
    """Экспортирует аномалии из последнего ансамблевого предсказания"""
    if data_dict is None:
        return None, "❌ Сначала загрузите данные!"

    if ensemble_results is None:
        return None, "❌ Сначала выполните ансамблевое предсказание (кнопка 'Выполнить ансамблевое предсказание')!"

    anomaly_mask = ensemble_results.get('anomaly_mask')
    if anomaly_mask is None or sum(anomaly_mask) == 0:
        return None, "⚠️ Аномалий не найдено! Нечего экспортировать."

    # Очищаем имя файла от недопустимых символов
    if not filename:
        filename = "anomalies"
    else:
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
        if not filename:
            filename = "anomalies"

    # Берём исходный датафрейм и фильтруем только аномалии
    original_df = data_dict['original']
    anomalies_df = original_df[anomaly_mask].copy()

    # Добавляем расширение .csv, если его нет
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Создаём файл во временной директории с заданным именем
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    # Сохраняем с включённым индексом
    anomalies_df.to_csv(file_path, index=True, index_label=original_df.index.name, encoding='utf-8-sig')

    return file_path, f"✅ Экспортировано {sum(anomaly_mask)} аномалий в файл: {filename}"


# ========== Создание интерфейса ==========
with gr.Blocks(title="Детектор аномалий", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Система обнаружения аномалий")
    gr.Markdown("> 💡 **Инструкция:** Загрузите данные → Выберите колонки → Запустите оптимизацию")

    data_state = gr.State()
    ensemble_results_state = gr.State(None)  # Состояние для хранения результатов ансамбля

    with gr.Tabs():
        # ===== Вкладка 1: Загрузка данных =====
        with gr.TabItem("📁 Шаг 1: Загрузка данных"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="CSV файл", file_types=[".csv"])
                    load_btn = gr.Button("Загрузить и нормализовать", variant="primary")
                with gr.Column():
                    info_output = gr.Textbox(label="Информация", lines=10)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Исходные данные (первые 5 строк)")
                    table_output = gr.Dataframe(label="Исходные данные", interactive=False)
                with gr.Column():
                    gr.Markdown("### Нормализованные данные (по выбранным колонкам)")
                    normalized_table_output = gr.Dataframe(label="Нормализованные данные", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Выберите колонки для обучения")
                    columns_selector = gr.CheckboxGroup(
                        choices=[],
                        label="Доступные числовые колонки",
                        interactive=True
                    )
                    columns_status = gr.Textbox(label="Статус выбора колонок", value="❌ Загрузите данные",
                                                interactive=False)

            load_btn.click(load_data, [file_input],
                           [data_state, info_output, table_output, normalized_table_output, columns_selector,
                            columns_status])
            file_input.change(load_data, [file_input],
                              [data_state, info_output, table_output, normalized_table_output, columns_selector,
                               columns_status])
            columns_selector.change(update_selected_columns, [data_state, columns_selector],
                                    [data_state, columns_status, normalized_table_output])

        # ===== Вкладка 2: Оптимизация (Optuna) =====
        with gr.TabItem("🎯 Шаг 2: Оптимизация гиперпараметров"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Выбор алгоритма")
                    optuna_algorithm = gr.Radio(
                        choices=["OneClassSVM", "IsolationForest", "LocalOutlierFactor"],
                        label="Алгоритм для оптимизации",
                        value="OneClassSVM"
                    )

                    gr.Markdown("### Параметры оптимизации")
                    n_trials = gr.Slider(10, 200, 50, 10, label="Количество попыток (n_trials)")

                    gr.Markdown("### Границы доли аномалий (borders)")
                    with gr.Row():
                        borders_low = gr.Number(value=0.03, label="Нижняя граница", precision=2, step=0.01)
                        borders_high = gr.Number(value=0.05, label="Верхняя граница", precision=2, step=0.01)
                    penalty = gr.Number(value=0, label="Штраф при выходе за границы", precision=0)

                    gr.Markdown("### Гиперпараметры для перебора")

                    with gr.Group(visible=True) as optuna_svm_params:
                        gr.Markdown("#### OneClass SVM")
                        with gr.Row():
                            svm_nu_min = gr.Number(value=0.03, label="nu min", precision=2, step=0.01)
                            svm_nu_max = gr.Number(value=0.05, label="nu max", precision=2, step=0.01)
                        svm_nu_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                    with gr.Group(visible=False) as optuna_iforest_params:
                        gr.Markdown("#### Isolation Forest")
                        with gr.Row():
                            iforest_n_min = gr.Number(value=50, label="n_estimators min", precision=0)
                            iforest_n_max = gr.Number(value=300, label="n_estimators max", precision=0)
                        iforest_n_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                        with gr.Row():
                            iforest_cont_min = gr.Number(value=0.03, label="contamination min", precision=2, step=0.01)
                            iforest_cont_max = gr.Number(value=0.05, label="contamination max", precision=2, step=0.01)
                        iforest_cont_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                        with gr.Row():
                            iforest_samples_min = gr.Number(value=0.5, label="max_samples min", precision=2, step=0.05)
                            iforest_samples_max = gr.Number(value=1.0, label="max_samples max", precision=2, step=0.05)
                        iforest_samples_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                    with gr.Group(visible=False) as optuna_lof_params:
                        gr.Markdown("#### Local Outlier Factor")
                        with gr.Row():
                            lof_neighbors_min = gr.Number(value=5, label="n_neighbors min", precision=0)
                            lof_neighbors_max = gr.Number(value=50, label="n_neighbors max", precision=0)
                        lof_neighbors_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                        with gr.Row():
                            lof_cont_min = gr.Number(value=0.03, label="contamination min", precision=2, step=0.01)
                            lof_cont_max = gr.Number(value=0.05, label="contamination max", precision=2, step=0.01)
                        lof_cont_log = gr.Checkbox(label="Логарифмический масштаб", value=False)

                    optimize_btn = gr.Button("🚀 Запустить оптимизацию", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Результаты оптимизации")
                    optuna_result = gr.Textbox(label="Детали", lines=12)
                    best_params_display = gr.Textbox(label="Лучшие параметры", lines=6)
                    best_score = gr.Number(label="Лучшая метрика", interactive=False)
                    optuna_logs = gr.Textbox(label="Логи Optuna (пошагово)", lines=15, interactive=False)


            def update_optuna_visibility(algo):
                return [
                    gr.update(visible=(algo == "OneClassSVM")),
                    gr.update(visible=(algo == "IsolationForest")),
                    gr.update(visible=(algo == "LocalOutlierFactor"))
                ]


            optuna_algorithm.change(update_optuna_visibility, optuna_algorithm,
                                    [optuna_svm_params, optuna_iforest_params, optuna_lof_params])


            def update_logs(logs):
                return logs


            def run_optimization_wrapper(
                    data_dict, algorithm, n_trials, bl, bh, p,
                    svm_nu_min, svm_nu_max, svm_nu_log,
                    if_n_min, if_n_max, if_n_log,
                    if_c_min, if_c_max, if_c_log,
                    if_s_min, if_s_max, if_s_log,
                    lof_n_min, lof_n_max, lof_n_log,
                    lof_c_min, lof_c_max, lof_c_log
            ):
                if bl >= bh:
                    return "❌ Ошибка: нижняя граница borders должна быть меньше верхней!", None, None, ""

                if algorithm == "OneClassSVM":
                    params = {
                        'nu': ((svm_nu_min, svm_nu_max), float, svm_nu_log)
                    }
                    cls = OneClassSVM

                elif algorithm == "IsolationForest":
                    params = {
                        'n_estimators': ((int(if_n_min), int(if_n_max)), int, if_n_log),
                        'contamination': ((if_c_min, if_c_max), float, if_c_log),
                        'max_samples': ((if_s_min, if_s_max), float, if_s_log),
                        'n_jobs': -1,
                        'random_state': 42
                    }
                    cls = IsolationForest

                else:
                    params = {
                        'n_neighbors': ((int(lof_n_min), int(lof_n_max)), int, lof_n_log),
                        'contamination': ((lof_c_min, lof_c_max), float, lof_c_log),
                        'n_jobs': -1
                    }
                    cls = LocalOutlierFactor

                return run_optimization(
                    data_dict, cls, int(n_trials), bl, bh, p, update_logs, **params
                )


            optimize_btn.click(
                run_optimization_wrapper,
                inputs=[data_state, optuna_algorithm, n_trials, borders_low, borders_high, penalty,
                        svm_nu_min, svm_nu_max, svm_nu_log,
                        iforest_n_min, iforest_n_max, iforest_n_log,
                        iforest_cont_min, iforest_cont_max, iforest_cont_log,
                        iforest_samples_min, iforest_samples_max, iforest_samples_log,
                        lof_neighbors_min, lof_neighbors_max, lof_neighbors_log,
                        lof_cont_min, lof_cont_max, lof_cont_log],
                outputs=[optuna_result, best_params_display, best_score, optuna_logs]
            )

        # ===== Вкладка 3: Предсказания =====
        with gr.TabItem("📊 Шаг 3: Предсказания"):
            with gr.Row():
                with gr.Column():
                    predict_selector = gr.Dropdown([], label="Доступные модели")
                    refresh_predict = gr.Button("🔄 Обновить список", variant="secondary")
                    predict_status = gr.Textbox("📭 Нет обученных моделей", label="Статус")
                    predict_btn = gr.Button("🔮 Получить предсказания", variant="primary")
                with gr.Column():
                    predict_result = gr.Textbox(label="Результаты предсказания", lines=25)

            refresh_predict.click(refresh_predict_models, None, [predict_selector, predict_status])
            predict_btn.click(get_predictions_from_selected_model, [data_state, predict_selector], predict_result)

        # ===== Вкладка 4: Визуализация =====
        with gr.TabItem("📈 Шаг 4: Визуализация"):
            with gr.Row():
                with gr.Column(scale=1):
                    plot_selector = gr.Dropdown([], label="Доступные модели")
                    refresh_plot = gr.Button("🔄 Обновить список", variant="secondary")
                    plot_status = gr.Textbox("📭 Нет обученных моделей", label="Статус")

                    gr.Markdown("### Тип визуализации")
                    plot_type = gr.Radio(
                        choices=["PCA", "Score Plot", "SHAP Summary", "SHAP Decision", "SHAP Heatmap"],
                        label="Выберите тип графика",
                        value="PCA"
                    )

                    # Группа для параметров SHAP Summary (скрыта по умолчанию)
                    with gr.Group(visible=False) as shap_summary_group:
                        summary_plot_type = gr.Radio(
                            choices=["dot", "bar", "violin"],
                            label="Тип summary plot",
                            value="dot"
                        )

                    plot_btn = gr.Button("📈 Построить график", variant="primary")

                with gr.Column(scale=1):
                    plot_output = gr.Plot(label="Визуализация")
                    plot_result = gr.Textbox(label="Результат", lines=3)

            refresh_plot.click(refresh_plot_models, None, [plot_selector, plot_status])


            def update_summary_visibility(plot_type_val):
                return gr.update(visible=(plot_type_val == "SHAP Summary"))


            plot_type.change(update_summary_visibility, plot_type, shap_summary_group)


            def create_plot_wrapper(data_dict, selected_model, plot_type_val, summary_type):
                if plot_type_val == "PCA":
                    return create_pca_plot(data_dict, selected_model)
                elif plot_type_val == "Score Plot":
                    return create_score_plot(data_dict, selected_model)
                else:
                    shap_type = {
                        "SHAP Summary": "summary",
                        "SHAP Decision": "decision",
                        "SHAP Heatmap": "heatmap"
                    }.get(plot_type_val, "summary")
                    return create_shap_plot(data_dict, selected_model, shap_type, summary_type)


            plot_btn.click(
                create_plot_wrapper,
                [data_state, plot_selector, plot_type, summary_plot_type],
                [plot_output, plot_result]
            )

        # ===== Вкладка 5: Управление кэшем =====
        with gr.TabItem("🗂️ Шаг 5: Кэш"):
            with gr.Row():
                with gr.Column():
                    show_btn = gr.Button("👁️ Показать все модели", variant="secondary")
                    cache_info = gr.Textbox(label="Информация о кэше", lines=15)
                    clear_btn = gr.Button("🗑️ Очистить кэш", variant="stop")
                    clear_result = gr.Textbox(label="Результат очистки", lines=3)

            show_btn.click(show_cached_models, None, cache_info)
            clear_btn.click(clear_cache, None, clear_result)

        # ===== Вкладка 6: Ансамбль аномалий =====
        with gr.TabItem("🎯 Шаг 6: Ансамбль аномалий"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Выберите модели для ансамбля")
                    gr.Markdown("> 💡 Алгоритмы, которые будут участвовать в голосовании")

                    available_models = gr.CheckboxGroup(
                        choices=[],
                        label="Доступные модели",
                        interactive=True
                    )

                    refresh_ensemble_btn = gr.Button("🔄 Обновить список моделей", variant="secondary")

                    gr.Markdown("### Параметры голосования")
                    p_threshold = gr.Slider(
                        0.0, 1.0, 0.5, 0.05,
                        label="Порог голосования (p)",
                        info="Объект считается аномалией, если доля алгоритмов, определивших его как аномалию, >= p"
                    )

                    gr.Markdown("### Визуализация распределений аномалий")
                    show_numerical = gr.Checkbox(label="Показать гистограммы числовых признаков", value=True)
                    show_categorical = gr.Checkbox(label="Показать столбчатые диаграммы категориальных признаков",
                                                   value=True)

                    ensemble_predict_btn = gr.Button("🔮 Выполнить ансамблевое предсказание", variant="primary",
                                                     size="lg")

                    gr.Markdown("### Экспорт результатов")
                    export_filename = gr.Textbox(
                        label="Имя файла для экспорта",
                        placeholder="anomalies (без расширения или с .csv)",
                        value="anomalies"
                    )
                    export_btn = gr.Button("📥 Скачать аномалии (CSV)", variant="secondary")
                    export_file = gr.File(label="Скачать CSV", visible=True)
                    export_status = gr.Textbox(label="Статус экспорта", lines=2, interactive=False)

                with gr.Column(scale=1):
                    ensemble_result = gr.Textbox(label="Результаты ансамбля", lines=12)
                    ensemble_stats = gr.Textbox(label="Статистика", lines=5)

            # PCA график - первый
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Визуализация аномалий (PCA)")
                    ensemble_plot = gr.Plot(label="Аномалии по результатам голосования")
                    ensemble_plot_status = gr.Textbox(label="Статус PCA", lines=2, interactive=False)

            # Числовые распределения - второй блок (под PCA)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Распределение числовых признаков среди аномалий")
                    numerical_plot = gr.Plot(label="Гистограммы числовых признаков")

            # Категориальные распределения - третий блок (под числовыми)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Распределение категориальных признаков среди аномалий")
                    categorical_plot = gr.Plot(label="Столбчатые диаграммы категориальных признаков")

            refresh_ensemble_btn.click(refresh_ensemble_models, None, [available_models, ensemble_stats])

            ensemble_predict_btn.click(
                ensemble_prediction,
                inputs=[data_state, available_models, p_threshold, show_numerical, show_categorical],
                outputs=[ensemble_result, ensemble_stats, ensemble_plot, numerical_plot, categorical_plot,
                         ensemble_plot_status, ensemble_results_state]
            )

            # Экспорт аномалий с пользовательским именем (использует сохранённые результаты)
            export_btn.click(
                export_anomalies_to_csv,
                inputs=[data_state, ensemble_results_state, export_filename],
                outputs=[export_file, export_status]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)