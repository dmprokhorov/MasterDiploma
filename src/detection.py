# streamlit_anomaly_detector.py - Детектор аномалий с оптимизацией гиперпараметров и SHAP визуализациями
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
import tempfile
import os
import re
import time
import math
import warnings

warnings.filterwarnings('ignore')

# Настройка графиков
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Инициализация состояния сессии
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'shap_explanations' not in st.session_state:
    st.session_state.shap_explanations = {}
if 'ensemble_results' not in st.session_state:
    st.session_state.ensemble_results = None
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None


def clear_all_cache():
    """Полностью очищает все кэши"""
    st.session_state.trained_models = {}
    st.session_state.shap_explanations = {}
    st.session_state.ensemble_results = None
    st.session_state.data_dict = None
    st.session_state.loaded = False
    return "Все кэши очищены"


# ==================== МЕТРИКА КАЧЕСТВА ====================
def anomaly_isolation_ratio_score(X, labels):
    """Метрика качества для аномалий - отношение расстояния аномалий от центра к компактности нормальных"""
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


def add_score_samples(model, data, indices, attribute='min_samples'):
    """Добавляет метод score_samples для DBSCAN"""
    nn = NearestNeighbors(n_neighbors=getattr(model, attribute))
    nn.fit(data.loc[~data.index.isin(indices)])
    setattr(model, 'score_samples', lambda y: nn.kneighbors(y)[0].mean(axis=1))
    return model


# ==================== ЕДИНАЯ ЦЕЛЕВАЯ ФУНКЦИЯ ДЛЯ OPTUNA ====================
def objective(trial, X, cls, borders=(0.03, 0.05), penalty=0, make_negative=False, **params):
    """Целевая функция для Optuna - единая для всех алгоритмов"""
    trial_params = {}
    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            if len(value) >= 3 and isinstance(value[2], bool):
                log = value[2]
            else:
                log = False

            if value[1] == float:
                trial_params[key] = trial.suggest_float(key, value[0][0], value[0][1], log=log)
            elif value[1] == int:
                trial_params[key] = trial.suggest_int(key, value[0][0], value[0][1], log=log)
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


# ==================== SHAP ФУНКЦИИ ====================
def get_shap_explained(attribute, data, indices, type_explainer='Tree', convert=False, **explainer_params):
    """Возвращает SHAP объяснения для аномалий"""
    feature_names = list(data.columns)
    outliers = data.iloc[indices].values

    if convert:
        data = torch.FloatTensor(data.values)
        outliers = torch.FloatTensor(outliers)

    explainer = getattr(shap, f'{type_explainer}Explainer')(attribute, data, **explainer_params)
    explained = explainer(outliers)
    explained.feature_names = feature_names
    return explained


def shap_summary_plot(explained, data, indices, plot_type='dot'):
    """Строит summary plot для SHAP"""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(explained.values, data.iloc[indices], plot_type=plot_type,
                      feature_names=explained.feature_names, max_display=15, show=False)
    plt.tight_layout()
    return plt.gcf()


def shap_decision_plot(explained, data, indices):
    """Строит decision plot для SHAP"""
    plt.figure(figsize=(12, 6))
    base_value = explained.base_values.mean() if hasattr(explained,
                                                         'base_values') and explained.base_values is not None else 0
    shap.decision_plot(base_value, explained.values, data.iloc[indices].values,
                       feature_names=data.columns.tolist(), show=False)
    plt.tight_layout()
    return plt.gcf()


def shap_heatmap_plot(explained):
    """Строит heatmap для SHAP"""
    plt.figure(figsize=(12, 6))
    if explained.values.ndim > 2:
        explained.values = explained.values.mean(axis=-1)
    shap.plots.heatmap(explained, show=False)
    plt.tight_layout()
    return plt.gcf()


# ==================== ФУНКЦИИ ВИЗУАЛИЗАЦИИ ====================
def create_pca_plot(data, predictions, model_name, indices=None):
    """Строит график PCA для визуализации аномалий с подписями"""
    fig, ax = plt.subplots(figsize=(12, 8))

    n_anomalies = sum(predictions == -1)
    n_normal = len(predictions) - n_anomalies

    if data.shape[1] >= 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        normal = data_2d[predictions != -1]
        anomalies = data_2d[predictions == -1]

        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_labels = data.index[anomaly_indices].tolist() if indices is None else indices[anomaly_indices].tolist()

        # Рисуем нормальные точки (одна легенда)
        ax.scatter(normal[:, 0], normal[:, 1], c='blue', label=f'Нормальные ({n_normal})', alpha=0.4, s=30)

        # Рисуем все аномалии без легенды для каждой
        for i, (x, y, label) in enumerate(zip(anomalies[:, 0], anomalies[:, 1], anomaly_labels)):
            ax.scatter(x, y, c='red', marker='x', s=100, linewidths=2, zorder=5)
            ax.annotate(str(label), (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, color='darkred', weight='bold')

        ax.scatter([], [], c='red', marker='x', s=100, linewidths=2, label=f'Аномалии ({n_anomalies})')

        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
    else:
        ax.plot(predictions, 'o-', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.fill_between(range(len(predictions)), -1, 1, where=(predictions == -1), color='red', alpha=0.3)

        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_labels = data.index[anomaly_indices].tolist() if indices is None else indices[anomaly_indices].tolist()

        for idx, label in zip(anomaly_indices, anomaly_labels):
            ax.annotate(str(label), (idx, -0.5), fontsize=8, alpha=0.8,
                        color='darkred', rotation=45, ha='right')

        # Легенда для одномерного графика
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                   label=f'Нормальные ({n_normal})'),
            Line2D([0], [0], marker='x', color='red', markersize=8, label=f'Аномалии ({n_anomalies})')
        ]
        ax.legend(handles=legend_elements)

        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Предсказание (1=норма, -1=аномалия)')

    ax.set_title(f'{model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scores(model, data, indices, method='score_samples'):
    """Строит точечную диаграмму значений выбросности объектов"""
    fig, ax = plt.subplots(figsize=(14, 6))

    scores = pd.Series(getattr(model, method)(data), index=data.index)

    ax.scatter(indices, scores.iloc[indices], color='red', label='Аномалии', s=50, alpha=0.7)
    normal_indices = np.delete(range(data.shape[0]), indices)
    ax.scatter(normal_indices, scores.iloc[normal_indices], color='blue', label='Норма', s=30, alpha=0.5)

    ax.set_xlabel('Индекс объекта', fontsize=12)
    ax.set_ylabel('Значение "выбросности" (score)', fontsize=12)
    ax.set_title('Точечная диаграмма значений "выбросности" объектов', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_numerical_distributions(data, indices, columns):
    """Строит гистограммы для численных признаков аномалий"""
    if len(columns) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Нет числовых признаков", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    ncols = min(int(math.ceil(len(columns) ** 0.5)), 4)
    nrows = max(1, math.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(data.iloc[indices][col], kde=True, color='red', alpha=0.6, ax=axes[i])
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel(None)

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Распределение числовых признаков среди аномалий (n={len(indices)})', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_categorical_distributions(data, indices, columns):
    """Строит столбчатые диаграммы для категориальных признаков аномалий"""
    if len(columns) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Нет категориальных признаков", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    ncols = min(int(math.ceil(len(columns) ** 0.5)), 4)
    nrows = max(1, math.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for i, col in enumerate(columns):
        if i < len(axes):
            value_counts = data.iloc[indices][col].value_counts()
            colors = sns.color_palette('rocket', len(value_counts))
            bars = axes[i].bar(range(len(value_counts)), value_counts.values, color=colors, alpha=0.7)
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel('Количество')
            for bar, val in zip(bars, value_counts.values):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val), ha='center',
                             va='bottom', fontsize=8)

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Распределение категориальных признаков среди аномалий (n={len(indices)})', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ==================== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ====================
def load_cluster_data(file):
    """Загружает данные кластера"""
    if file is None:
        return None, "Файл не выбран", None

    try:
        df = pd.read_csv(file, index_col=0)

        exclude_cols = ['cluster']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = [col for col in numeric_cols if col not in exclude_cols]

        data_dict = {
            'original': df,
            'normalized': None,
            'scaler': None,
            'all_columns': available_cols,
            'selected_columns': available_cols.copy()
        }

        return data_dict, f"Загружен файл: {df.shape[0]} строк, {len(available_cols)} числовых колонок", df.head()

    except Exception as e:
        return None, f"Ошибка загрузки: {str(e)}", None


def normalize_data(data_dict, selected_columns):
    """Нормализует выбранные колонки"""
    if data_dict is None or not selected_columns:
        return data_dict

    df = data_dict['original']
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[selected_columns])
    normalized_df = pd.DataFrame(normalized_data, columns=selected_columns, index=df.index)

    data_dict['normalized'] = normalized_df
    data_dict['scaler'] = scaler
    data_dict['selected_columns'] = selected_columns

    return data_dict


# ==================== ФУНКЦИИ ОПТИМИЗАЦИИ ====================
def run_optimization(data_dict, algorithm, params_config, log_placeholder):
    """Запускает оптимизацию гиперпараметров с обновляющимся логом"""
    if data_dict is None:
        return "Сначала загрузите данные!", None, 0

    selected_columns = data_dict.get('selected_columns', [])
    if not selected_columns:
        return "Выберите колонки для обучения!", None, 0

    X = data_dict['normalized'][selected_columns].copy()

    if algorithm in ["AutoEncoder", "KNN", "COPOD", "ECOD"]:
        make_negative = True
    else:
        make_negative = False

    if algorithm == "OneClassSVM":
        params = {
            'nu': ((params_config['nu_min'], params_config['nu_max']), float, params_config['nu_log']),
            'kernel': 'rbf'
        }
        cls = OneClassSVM
    elif algorithm == "IsolationForest":
        params = {
            'n_estimators': ((params_config['n_min'], params_config['n_max']), int, params_config['n_log']),
            'max_samples': ((params_config['samples_min'], params_config['samples_max']), float,
                            params_config['samples_log']),
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'n_jobs': -1,
            'random_state': 42
        }
        cls = IsolationForest
    elif algorithm == "LocalOutlierFactor":
        params = {
            'n_neighbors': ((params_config['n_min'], params_config['n_max']), int, params_config['n_log']),
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'metric': params_config['metric'],
            'n_jobs': -1
        }
        cls = LocalOutlierFactor
    elif algorithm == "DBSCAN":
        params = {
            'eps': ((params_config['eps_min'], params_config['eps_max']), float, params_config['eps_log']),
            'min_samples': ((params_config['ms_min'], params_config['ms_max']), int, params_config['ms_log']),
            'metric': params_config['metric'],
            'n_jobs': -1
        }
        cls = DBSCAN
    elif algorithm == "AutoEncoder":
        params = {
            'lr': ((params_config['lr_min'], params_config['lr_max']), float, params_config['lr_log']),
            'epoch_num': ((params_config['ep_min'], params_config['ep_max']), int, params_config['ep_log']),
            'dropout_rate': ((params_config['dr_min'], params_config['dr_max']), float, params_config['dr_log']),
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'verbose': False,
            'random_state': 42
        }
        cls = AutoEncoder
    elif algorithm == "KNN":
        params = {
            'n_neighbors': ((params_config['nn_min'], params_config['nn_max']), int, params_config['nn_log']),
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'method': params_config['method'],
            'metric': params_config['metric'],
            'n_jobs': -1
        }
        cls = KNN
    elif algorithm == "COPOD":
        params = {
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'n_jobs': -1
        }
        cls = COPOD
    elif algorithm == "ECOD":
        params = {
            'contamination': ((params_config['cont_min'], params_config['cont_max']), float, params_config['cont_log']),
            'n_jobs': -1
        }
        cls = ECOD

    study = optuna.create_study(direction='maximize', study_name=f'Optuna_{algorithm}')

    best_value = -float('inf')
    best_trial_number = None
    best_params_str = ""

    def objective_wrapper(trial):
        nonlocal best_value, best_trial_number, best_params_str
        result = objective(
            trial, X, cls,
            (params_config['borders_low'], params_config['borders_high']),
            params_config['penalty'],
            make_negative,
            **params
        )

        is_best = result > best_value
        if is_best:
            best_value = result
            best_trial_number = trial.number + 1
            best_params_str = f"метрика={best_value:.4f}"

        log_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ                   ║
║                        {algorithm}                               ║
╚══════════════════════════════════════════════════════════════════╝

ТЕКУЩАЯ ИТЕРАЦИЯ:
   Попытка: {trial.number + 1} / {params_config['n_trials']}
   Метрика: {result:.4f}
   {' НОВЫЙ ЛУЧШИЙ!' if is_best else ''}

ПАРАМЕТРЫ ТЕКУЩЕЙ ПОПЫТКИ:
{chr(10).join([f'   {k}: {v}' for k, v in trial.params.items()])}

ЛУЧШИЙ РЕЗУЛЬТАТ:
   Попытка #{best_trial_number if best_trial_number else 'Н/Д'}
   {best_params_str}

{'─' * 70}
"""
        log_placeholder.code(log_text, language="text")
        time.sleep(0.05)

        return result

    study.optimize(objective_wrapper, n_trials=params_config['n_trials'], show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value

    final_log = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОПТИМИЗАЦИЯ ЗАВЕРШЕНА                         ║
╚══════════════════════════════════════════════════════════════════╝

ЛУЧШИЕ ПАРАМЕТРЫ:
{chr(10).join([f'   {k}: {v}' for k, v in best_params.items()])}

ЛУЧШЕЕ ЗНАЧЕНИЕ МЕТРИКИ: {best_value:.4f}

{'─' * 70}
"""
    log_placeholder.code(final_log, language="text")

    final_model = cls(**best_params)
    predictions = final_model.fit_predict(X)
    if make_negative:
        predictions *= -1

    n_anomalies = sum(predictions == -1)
    anomaly_indices = np.where(predictions == -1)[0].tolist()
    score = anomaly_isolation_ratio_score(X, predictions)

    if algorithm == "DBSCAN":
        final_model = add_score_samples(final_model, X, anomaly_indices)

    novelty_model = None
    if algorithm == "LocalOutlierFactor":
        novelty_model = LocalOutlierFactor(**best_params, novelty=True, n_jobs=-1)
        novelty_model.fit(X)

    st.session_state.trained_models[algorithm] = {
        'model': final_model,
        'novelty_model': novelty_model,
        'predictions': predictions,
        'params': best_params,
        'columns': selected_columns.copy(),
        'anomaly_indices': anomaly_indices,
        'score': score,
        'n_anomalies': n_anomalies
    }

    if algorithm in st.session_state.shap_explanations:
        del st.session_state.shap_explanations[algorithm]

    result_text = f"""
ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!

ЛУЧШИЕ ПАРАМЕТРЫ:
{chr(10).join([f'   {k}: {v}' for k, v in best_params.items()])}

ЛУЧШЕЕ ЗНАЧЕНИЕ МЕТРИКИ: {score:.4f}
НАЙДЕНО АНОМАЛИЙ: {n_anomalies} ({n_anomalies / len(X) * 100:.1f}%)
ИНДЕКСЫ АНОМАЛИЙ (ПЕРВЫЕ 20): {anomaly_indices[:20]}"""

    return result_text, best_params, score


# ==================== ФУНКЦИИ ВИЗУАЛИЗАЦИИ МОДЕЛЕЙ ====================
def get_trained_models_list():
    """Возвращает список обученных моделей"""
    if not st.session_state.trained_models:
        return []
    return list(st.session_state.trained_models.keys())


def create_plot(data_dict, model_name, plot_type, summary_plot_type='dot'):
    """Создает график для выбранной модели"""
    if data_dict is None:
        return None, "Сначала загрузите данные!"

    if model_name not in st.session_state.trained_models:
        return None, f"Модель {model_name} не найдена!"

    model_data = st.session_state.trained_models[model_name]
    X = data_dict['normalized'][model_data['columns']].copy()
    predictions = model_data['predictions']
    anomaly_indices = model_data['anomaly_indices']

    if plot_type == "PCA":
        fig = create_pca_plot(X, predictions, model_name, X.index)
        return fig, None
    elif plot_type == "Score Plot":
        if len(anomaly_indices) == 0:
            return None, "Нет аномалий для отображения!"

        model = model_data['model']
        if model_name == 'LocalOutlierFactor' and model_data.get('novelty_model') is not None:
            model = model_data['novelty_model']

        method = 'score_samples' if hasattr(model, 'score_samples') else 'decision_function'
        fig = plot_scores(model, X, anomaly_indices, method)

        ax = fig.axes[0]
        anomaly_labels = X.index[anomaly_indices].tolist()
        for idx, label in zip(anomaly_indices[:20], anomaly_labels[:20]):
            ax.annotate(str(label), (idx, 0), xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.7, color='darkred')
        plt.tight_layout()

        return fig, None
    elif plot_type.startswith("SHAP"):
        if len(anomaly_indices) == 0:
            return None, "Нет аномалий для SHAP объяснений!"

        if model_name == "IsolationForest":
            type_explainer = 'Tree'
        elif model_name == "AutoEncoder":
            type_explainer = 'Deep'
        elif model_name in ("KNN", "ECOD", "COPOD"):
            type_explainer = 'Permutation'
        else:
            type_explainer = 'Kernel'

        if model_name in st.session_state.shap_explanations:
            explained = st.session_state.shap_explanations[model_name]
        else:
            if model_name == "LocalOutlierFactor" and model_data.get('novelty_model') is not None:
                attribute_model = model_data['novelty_model']
            else:
                attribute_model = model_data['model']

            if model_name == "IsolationForest":
                attribute = attribute_model
            elif model_name == "AutoEncoder":
                attribute = attribute_model.model
            elif hasattr(attribute_model, 'score_samples'):
                attribute = attribute_model.score_samples
            else:
                attribute = attribute_model.decision_function

            explained = get_shap_explained(attribute, X, anomaly_indices, type_explainer,
                                           convert=(type_explainer == 'Deep'))
            st.session_state.shap_explanations[model_name] = explained

        if plot_type == "SHAP Summary":
            fig = shap_summary_plot(explained, X, anomaly_indices, summary_plot_type)
            fig.axes[0].set_title(f'SHAP Summary - {model_name}\nАномалии: {len(anomaly_indices)} объектов',
                                  fontsize=12)
            return fig, None
        elif plot_type == "SHAP Decision":
            fig = shap_decision_plot(explained, X, anomaly_indices)
            fig.axes[0].set_title(f'SHAP Decision - {model_name}\nАномалии: {len(anomaly_indices)} объектов',
                                  fontsize=12)
            return fig, None
        elif plot_type == "SHAP Heatmap":
            fig = shap_heatmap_plot(explained)
            return fig, None

    return None, "Неизвестный тип графика"


# ==================== ФУНКЦИИ АНСАМБЛЯ ====================
def ensemble_prediction(data_dict, selected_models, threshold, show_numerical, show_categorical):
    """Выполняет ансамблевое предсказание"""
    if data_dict is None:
        return "Сначала загрузите данные!", "", None, None, None

    if not selected_models:
        return "Выберите хотя бы одну модель!", "", None, None, None

    all_predictions = []
    for model_name in selected_models:
        model_data = st.session_state.trained_models[model_name]
        all_predictions.append(model_data['predictions'])

    all_predictions = np.array(all_predictions)
    n_models = len(all_predictions)
    n_samples = all_predictions.shape[1]

    anomaly_votes = np.sum(all_predictions == -1, axis=0)
    anomaly_ratio = anomaly_votes / n_models

    ensemble_labels = np.where(anomaly_ratio >= threshold, -1, 1)
    n_anomalies = sum(ensemble_labels == -1)
    anomaly_indices = np.where(ensemble_labels == -1)[0].tolist()

    ensemble_results = {
        'anomaly_mask': ensemble_labels == -1,
        'anomaly_indices': anomaly_indices,
        'selected_models': selected_models,
        'threshold': threshold,
        'n_anomalies': n_anomalies,
        'n_models': n_models,
        'n_samples': n_samples,
        'anomaly_votes': anomaly_votes,
        'anomaly_ratio': anomaly_ratio
    }
    st.session_state.ensemble_results = ensemble_results

    first_model = st.session_state.trained_models[selected_models[0]]
    X = data_dict['normalized'][first_model['columns']].copy()

    fig_pca, ax = plt.subplots(figsize=(10, 5))
    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(X)
        normal = data_2d[ensemble_labels == 1]
        anomalies = data_2d[ensemble_labels == -1]

        anomaly_indices_full = np.where(ensemble_labels == -1)[0]
        anomaly_labels = X.index[anomaly_indices_full].tolist()

        ax.scatter(normal[:, 0], normal[:, 1], c='blue', label=f'Нормальные ({n_samples - n_anomalies})', alpha=0.4,
                   s=30)

        for i, (x, y, label) in enumerate(zip(anomalies[:, 0], anomalies[:, 1], anomaly_labels)):
            ax.scatter(x, y, c='red', marker='x', s=120, linewidths=2, zorder=5)
            ax.annotate(str(label), (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.9, color='darkred', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        ax.scatter([], [], c='red', marker='x', s=100, linewidths=2, label=f'Аномалии ({n_anomalies})')

        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
    else:
        ax.plot(ensemble_labels, 'o-', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.fill_between(range(len(ensemble_labels)), -1, 1, where=(ensemble_labels == -1), color='red', alpha=0.3)

        anomaly_indices_full = np.where(ensemble_labels == -1)[0]
        anomaly_labels = X.index[anomaly_indices_full].tolist()

        for idx, label in zip(anomaly_indices_full, anomaly_labels):
            ax.annotate(str(label), (idx, -0.5), fontsize=9, alpha=0.9,
                        color='darkred', rotation=45, ha='right')

        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Предсказание (1=норма, -1=аномалия)')

    ax.set_title(f'Ансамбль ({n_models} моделей), порог p={threshold}\nАномалии с подписями (логины школ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_numerical = None
    fig_categorical = None
    original_df = data_dict['original']

    if n_anomalies > 0:
        if show_numerical:
            numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
            fig_numerical = plot_numerical_distributions(original_df, anomaly_indices, numerical_cols)
        if show_categorical:
            categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns.tolist()
            fig_categorical = plot_categorical_distributions(original_df, anomaly_indices, categorical_cols)
    else:
        empty_fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет аномалий для визуализации", ha='center', va='center', fontsize=14)
        ax.axis('off')
        if show_numerical:
            fig_numerical = empty_fig
        if show_categorical:
            fig_categorical = empty_fig

    return fig_pca, fig_numerical, fig_categorical


def export_anomalies(data_dict, filename):
    """Экспортирует аномалии в CSV"""
    if data_dict is None:
        return None, "Сначала загрузите данные!"

    if st.session_state.ensemble_results is None:
        return None, "Сначала выполните ансамблевое предсказание!"

    anomaly_mask = st.session_state.ensemble_results['anomaly_mask']
    if sum(anomaly_mask) == 0:
        return None, "Аномалий не найдено!"

    if not filename:
        filename = "anomalies"
    else:
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
        if not filename:
            filename = "anomalies"

    anomalies_df = data_dict['original'][anomaly_mask].copy()

    if not filename.endswith('.csv'):
        filename += '.csv'

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    anomalies_df.to_csv(file_path, index=True, encoding='utf-8-sig')

    return file_path, f"Экспортировано {sum(anomaly_mask)} аномалий"


# ==================== ИНТЕРФЕЙС STREAMLIT ====================
st.set_page_config(page_title="Детектор аномалий", layout="wide")

st.title("Обнаружение аномалий")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Загрузка данных",
    "Обучение моделей",
    "Визуализация",
    "Ансамбль аномалий",
    "Кэш моделей"
])

# ========== Вкладка 1: Загрузка данных ==========
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("CSV файл (кластер)", type=["csv"], key="file_uploader")

        if st.button("Загрузить данные", type="primary", key="load_btn"):
            if st.session_state.current_file_name != uploaded_file.name:
                clear_all_cache()
                st.session_state.current_file_name = uploaded_file.name if uploaded_file else None
                st.rerun()

            with st.spinner("Загрузка..."):
                data_dict, info, preview = load_cluster_data(uploaded_file)
                if data_dict is not None:
                    st.session_state.data_dict = data_dict
                    st.session_state.loaded = True
                    st.dataframe(preview, use_container_width=True)

    with col2:
        if st.session_state.loaded and st.session_state.data_dict is not None:
            st.markdown("### Выберите колонки для анализа")
            all_cols = st.session_state.data_dict['all_columns']

            prev_selected = st.session_state.data_dict.get('selected_columns', all_cols)
            selected_cols = st.multiselect("Числовые колонки", all_cols, default=prev_selected, key="col_selector")

            if selected_cols != prev_selected:
                st.session_state.trained_models = {}
                st.session_state.shap_explanations = {}
                st.session_state.ensemble_results = None

            if selected_cols:
                st.session_state.data_dict = normalize_data(st.session_state.data_dict, selected_cols)
                st.dataframe(st.session_state.data_dict['normalized'].head(), use_container_width=True)


# ========== Вкладка 2: Обучение моделей ==========
with tab2:
    if not st.session_state.loaded or st.session_state.data_dict is None or st.session_state.data_dict.get(
            'normalized') is None:
        st.warning("Сначала загрузите данные и выберите колонки на вкладке 1")
    else:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### Выбор алгоритма")
            algorithm = st.selectbox("Алгоритм для оптимизации", [
                "OneClassSVM", "IsolationForest", "LocalOutlierFactor", "DBSCAN",
                "AutoEncoder", "KNN", "COPOD", "ECOD"
            ], key="algo_selector")

            st.markdown("### Параметры оптимизации")
            n_trials = st.slider("Количество попыток", 10, 100, 30, 5, key="n_trials")

            st.markdown("### Границы доли аномалий")
            col_a, col_b = st.columns(2)
            with col_a:
                borders_low = st.number_input("Нижняя граница", 0.01, 0.2, 0.03, 0.01, key="borders_low")
            with col_b:
                borders_high = st.number_input("Верхняя граница", 0.02, 0.3, 0.05, 0.01, key="borders_high")
            penalty = st.number_input("Штраф", 0, 100, 0, key="penalty")

            st.markdown("### Гиперпараметры для перебора")

            # Параметры для OneClassSVM
            if algorithm == "OneClassSVM":
                with st.expander("Доля выбросов (nu)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: nu_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="nu_min", format="%.4f")
                    with col_b: nu_max = st.number_input("Максимум", 0.01, 0.3, 0.05, 0.01, key="nu_max", format="%.4f")
                    with col_c: nu_log = st.checkbox("Логарифмический масштаб", False, key="nu_log")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'nu_min': nu_min, 'nu_max': nu_max, 'nu_log': nu_log
                }

            # Параметры для IsolationForest
            elif algorithm == "IsolationForest":
                with st.expander("Количество деревьев (n_estimators)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: n_min = st.number_input("Минимум", 50, 500, 100, 10, key="n_min")
                    with col_b: n_max = st.number_input("Максимум", 100, 1000, 300, 50, key="n_max")
                    with col_c: n_log = st.checkbox("Логарифмический масштаб", False, key="n_log")

                with st.expander("Доля выборки (max_samples)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: samples_min = st.number_input("Минимум", 0.5, 0.9, 0.7, 0.05, key="samples_min")
                    with col_b: samples_max = st.number_input("Максимум", 0.8, 1.0, 1.0, 0.05, key="samples_max")
                    with col_c: samples_log = st.checkbox("Логарифмический масштаб", False, key="samples_log")

                with st.expander("Доля выбросов (contamination)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: cont_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="cont_min")
                    with col_b: cont_max = st.number_input("Максимум", 0.02, 0.3, 0.05, 0.01, key="cont_max")
                    with col_c: cont_log = st.checkbox("Логарифмический масштаб", False, key="cont_log")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'n_min': n_min, 'n_max': n_max, 'n_log': n_log,
                    'samples_min': samples_min, 'samples_max': samples_max, 'samples_log': samples_log,
                    'cont_min': cont_min, 'cont_max': cont_max, 'cont_log': cont_log
                }

            # Параметры для LocalOutlierFactor
            elif algorithm == "LocalOutlierFactor":
                with st.expander("Количество соседей (n_neighbors)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: n_min = st.number_input("Минимум", 5, 50, 10, 5, key="n_min")
                    with col_b: n_max = st.number_input("Максимум", 10, 100, 50, 10, key="n_max")
                    with col_c: n_log = st.checkbox("Логарифмический масштаб", False, key="n_log")

                with st.expander("Доля выбросов (contamination)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: cont_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="cont_min")
                    with col_b: cont_max = st.number_input("Максимум", 0.02, 0.3, 0.05, 0.01, key="cont_max")
                    with col_c: cont_log = st.checkbox("Логарифмический масштаб", False, key="cont_log")

                metric = st.selectbox("Метрика расстояния (metric)", ["minkowski", "euclidean", "manhattan"], key="metric")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'n_min': n_min, 'n_max': n_max, 'n_log': n_log,
                    'cont_min': cont_min, 'cont_max': cont_max, 'cont_log': cont_log,
                    'metric': metric
                }

            # Параметры для DBSCAN
            elif algorithm == "DBSCAN":
                with st.expander("Радиус окрестности (eps)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: eps_min = st.number_input("Минимум", 0.5, 1.0, 0.75, 0.05, key="eps_min")
                    with col_b: eps_max = st.number_input("Максимум", 1.0, 2.0, 1.5, 0.1, key="eps_max")
                    with col_c: eps_log = st.checkbox("Логарифмический масштаб", False, key="eps_log")

                with st.expander("Минимальное количество точек (min_samples)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: ms_min = st.number_input("Минимум", 5, 20, 7, 1, key="ms_min")
                    with col_b: ms_max = st.number_input("Максимум", 10, 30, 15, 2, key="ms_max")
                    with col_c: ms_log = st.checkbox("Логарифмический масштаб", False, key="ms_log")

                metric = st.selectbox("Метрика расстояния (metric)", ["euclidean", "manhattan", "cosine"], key="metric")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'eps_min': eps_min, 'eps_max': eps_max, 'eps_log': eps_log,
                    'ms_min': ms_min, 'ms_max': ms_max, 'ms_log': ms_log,
                    'metric': metric
                }

            # Параметры для AutoEncoder
            elif algorithm == "AutoEncoder":
                with st.expander("Скорость обучения (lr)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: lr_min = st.number_input("Минимум", 0.0001, 0.001, 0.0005, 0.0001, format="%.4f", key="lr_min")
                    with col_b: lr_max = st.number_input("Максимум", 0.001, 0.05, 0.01, 0.005, format="%.4f", key="lr_max")
                    with col_c: lr_log = st.checkbox("Логарифмический масштаб", True, key="lr_log")

                with st.expander("Количество эпох (epoch_num)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: ep_min = st.number_input("Минимум", 10, 50, 20, 5, key="ep_min")
                    with col_b: ep_max = st.number_input("Максимум", 20, 100, 50, 10, key="ep_max")
                    with col_c: ep_log = st.checkbox("Логарифмический масштаб", False, key="ep_log")

                with st.expander("Вероятность dropout (dropout_rate)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: dr_min = st.number_input("Минимум", 0.0, 0.2, 0.0, 0.05, key="dr_min")
                    with col_b: dr_max = st.number_input("Максимум", 0.1, 0.5, 0.3, 0.1, key="dr_max")
                    with col_c: dr_log = st.checkbox("Логарифмический масштаб", False, key="dr_log")

                with st.expander("Доля выбросов (contamination)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: cont_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="cont_min")
                    with col_b: cont_max = st.number_input("Максимум", 0.02, 0.3, 0.05, 0.01, key="cont_max")
                    with col_c: cont_log = st.checkbox("Логарифмический масштаб", False, key="cont_log")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'lr_min': lr_min, 'lr_max': lr_max, 'lr_log': lr_log,
                    'ep_min': ep_min, 'ep_max': ep_max, 'ep_log': ep_log,
                    'dr_min': dr_min, 'dr_max': dr_max, 'dr_log': dr_log,
                    'cont_min': cont_min, 'cont_max': cont_max, 'cont_log': cont_log
                }

            # Параметры для KNN
            elif algorithm == "KNN":
                with st.expander("Количество соседей (n_neighbors)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: nn_min = st.number_input("Минимум", 3, 15, 5, 1, key="nn_min")
                    with col_b: nn_max = st.number_input("Максимум", 10, 50, 15, 5, key="nn_max")
                    with col_c: nn_log = st.checkbox("Логарифмический масштаб", False, key="nn_log")

                with st.expander("Доля выбросов (contamination)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: cont_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="cont_min")
                    with col_b: cont_max = st.number_input("Максимум", 0.02, 0.3, 0.05, 0.01, key="cont_max")
                    with col_c: cont_log = st.checkbox("Логарифмический масштаб", False, key="cont_log")

                method = st.selectbox("Метод агрегации (method)", ["largest", "mean", "median"], key="method")
                metric = st.selectbox("Метрика расстояния (metric)", ["minkowski", "euclidean", "manhattan"], key="metric")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'nn_min': nn_min, 'nn_max': nn_max, 'nn_log': nn_log,
                    'cont_min': cont_min, 'cont_max': cont_max, 'cont_log': cont_log,
                    'method': method, 'metric': metric
                }

            # Параметры для COPOD/ECOD
            elif algorithm in ["COPOD", "ECOD"]:
                with st.expander("Доля выбросов (contamination)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: cont_min = st.number_input("Минимум", 0.01, 0.2, 0.03, 0.01, key="cont_min")
                    with col_b: cont_max = st.number_input("Максимум", 0.02, 0.3, 0.05, 0.01, key="cont_max")
                    with col_c: cont_log = st.checkbox("Логарифмический масштаб", False, key="cont_log")

                params_config = {
                    'n_trials': n_trials,
                    'borders_low': borders_low,
                    'borders_high': borders_high,
                    'penalty': penalty,
                    'cont_min': cont_min, 'cont_max': cont_max, 'cont_log': cont_log
                }

            train_btn = st.button("Запустить оптимизацию", type="primary", use_container_width=True, key="train_btn")

        with col2:
            st.markdown("### Логи оптимизации (обновляются в реальном времени)")
            log_placeholder = st.empty()
            result_placeholder = st.empty()

            if train_btn:
                with st.spinner(f"Оптимизация для {algorithm}..."):
                    result, best_params, score = run_optimization(
                        st.session_state.data_dict, algorithm, params_config, log_placeholder
                    )
                    result_placeholder.text_area("Результаты оптимизации", result, height=200, key="result_area")


# ========== Вкладка 3: Визуализация ==========
with tab3:
    if not st.session_state.trained_models:
        st.info("Сначала обучите модели на вкладке 2")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            models_list = get_trained_models_list()
            selected_model = st.selectbox("Выберите модель", models_list, key="model_select")

            plot_type = st.radio("Тип графика",
                                 ["PCA", "Score Plot", "SHAP Summary", "SHAP Decision", "SHAP Heatmap"],
                                 key="plot_type")

            summary_type = "dot"
            if plot_type == "SHAP Summary":
                summary_type = st.radio("Тип summary графика", ["dot", "bar", "violin"], key="summary_type")

            plot_btn = st.button("Построить график", type="primary", use_container_width=True, key="plot_btn")

        with col2:
            plot_placeholder = st.empty()

            if plot_btn:
                with st.spinner("Строим график..."):
                    fig, msg = create_plot(st.session_state.data_dict, selected_model, plot_type, summary_type)
                    if fig is not None:
                        plot_placeholder.pyplot(fig)
                    elif msg:
                        st.error(msg)


# ========== Вкладка 4: Ансамбль аномалий ==========
with tab4:
    if not st.session_state.trained_models:
        st.info("Сначала обучите модели на вкладке 2")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            models_list = get_trained_models_list()
            selected_models = st.multiselect("Выберите модели для ансамбля", models_list, key="ensemble_models")

            threshold = st.slider("Порог голосования (p)", 0.0, 1.0, 0.5, 0.05, key="threshold")

            st.markdown("### Визуализация распределений")
            show_numerical = st.checkbox("Показать числовые признаки", True, key="show_numerical")
            show_categorical = st.checkbox("Показать категориальные признаки", True, key="show_categorical")

            ensemble_btn = st.button("Выполнить ансамбль", type="primary", use_container_width=True, key="ensemble_btn")

            st.markdown("---")
            st.markdown("### Экспорт результатов")
            export_filename = st.text_input("Имя файла", "anomalies", key="export_filename")

            # Кнопка скачивания здесь
            if st.session_state.ensemble_results is not None:
                anomaly_mask = st.session_state.ensemble_results['anomaly_mask']
                if sum(anomaly_mask) > 0:
                    anomalies_df = st.session_state.data_dict['original'][anomaly_mask].copy()
                    csv_data = anomalies_df.to_csv(index=True, encoding='utf-8-sig')
                    st.download_button(
                        label="Скачать CSV",
                        data=csv_data,
                        file_name=f"{export_filename}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="export_btn"
                    )
                else:
                    st.info("Аномалий не найдено - нечего экспортировать")
            else:
                st.info("Сначала выполните ансамблевое предсказание")

        with st.container():
            st.markdown("### Визуализация аномалий (PCA)")
            ensemble_plot = st.empty()

            st.markdown("### Распределение числовых признаков")
            numerical_plot = st.empty()

            st.markdown("### Распределение категориальных признаков")
            categorical_plot = st.empty()

        if 'ensemble_fig_pca' not in st.session_state:
            st.session_state.ensemble_fig_pca = None
        if 'ensemble_fig_num' not in st.session_state:
            st.session_state.ensemble_fig_num = None
        if 'ensemble_fig_cat' not in st.session_state:
            st.session_state.ensemble_fig_cat = None

        if ensemble_btn:
            if not selected_models:
                st.error("Выберите хотя бы одну модель для ансамбля!")
            else:
                with st.spinner("Выполняется ансамбль..."):
                    # Изменяем вызов - убираем stats
                    fig_pca, fig_num, fig_cat = ensemble_prediction(
                        st.session_state.data_dict, selected_models, threshold, show_numerical, show_categorical
                    )

                    st.session_state.ensemble_fig_pca = fig_pca
                    st.session_state.ensemble_fig_num = fig_num
                    st.session_state.ensemble_fig_cat = fig_cat

                    ensemble_plot.pyplot(fig_pca)
                    if fig_num:
                        numerical_plot.pyplot(fig_num)
                    if fig_cat:
                        categorical_plot.pyplot(fig_cat)

                st.rerun()

        if st.session_state.ensemble_fig_pca is not None and not ensemble_btn:
            ensemble_plot.pyplot(st.session_state.ensemble_fig_pca)
        if st.session_state.ensemble_fig_num is not None and not ensemble_btn:
            numerical_plot.pyplot(st.session_state.ensemble_fig_num)
        if st.session_state.ensemble_fig_cat is not None and not ensemble_btn:
            categorical_plot.pyplot(st.session_state.ensemble_fig_cat)


# ========== Вкладка 5: Кэш моделей ==========
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Показать все модели", key="show_models_btn"):
            if not st.session_state.trained_models:
                st.info("Кэш пуст")
            else:
                for name, data in st.session_state.trained_models.items():
                    with st.expander(f"Модель {name}"):
                        st.write("**Гиперпараметры:**")
                        for k, v in data['params'].items():
                            # Переводим названия параметров на русский
                            param_names = {
                                'nu': 'Доля выбросов',
                                'n_estimators': 'Количество деревьев',
                                'max_samples': 'Доля выборки',
                                'contamination': 'Доля выбросов',
                                'n_neighbors': 'Количество соседей',
                                'metric': 'Метрика расстояния',
                                'eps': 'Радиус окрестности',
                                'min_samples': 'Мин. точек в окрестности',
                                'lr': 'Скорость обучения',
                                'epoch_num': 'Количество эпох',
                                'dropout_rate': 'Вероятность dropout',
                                'method': 'Метод агрегации'
                            }
                            param_name = param_names.get(k, k)
                            st.write(f"   - {param_name}: {v}")
                        st.write(f"**Колонки:** {data['columns'][:5]}...")
                        st.write(f"**Аномалий:** {data['n_anomalies']} ({data['n_anomalies'] / len(st.session_state.data_dict['normalized']) * 100:.1f}%)")
                        st.write(f"**Метрика:** {data['score']:.4f}")

        if st.button("Показать SHAP кэш", key="show_shap_btn"):
            if not st.session_state.shap_explanations:
                st.info("SHAP кэш пуст")
            else:
                for name in st.session_state.shap_explanations.keys():
                    st.write(f"SHAP значения для: {name}")

    with col2:
        if st.button("Очистить кэш моделей", type="secondary", key="clear_models_btn"):
            st.session_state.trained_models = {}
            st.session_state.shap_explanations = {}
            st.session_state.ensemble_results = None
            st.success("Кэш моделей очищен")

        if st.button("Очистить SHAP кэш", type="secondary", key="clear_shap_btn"):
            st.session_state.shap_explanations = {}
            st.success("SHAP кэш очищен")