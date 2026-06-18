# learning_deepseek7.py - Исправленная версия с уникальными ключами
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import optuna
from optuna import create_study
import tempfile
import os
import time
import shap
import warnings

warnings.filterwarnings('ignore')

# Настройка графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Инициализация состояния сессии
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'predictions_cache' not in st.session_state:
    st.session_state.predictions_cache = {}
if 'shap_cache' not in st.session_state:
    st.session_state.shap_cache = {}
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = None
if 'X1' not in st.session_state:
    st.session_state.X1 = None
if 'y1' not in st.session_state:
    st.session_state.y1 = None
if 'X2' not in st.session_state:
    st.session_state.X2 = None
if 'y2' not in st.session_state:
    st.session_state.y2 = None
if 'split_done' not in st.session_state:
    st.session_state.split_done = False
if 'show_cat_plot' not in st.session_state:
    st.session_state.show_cat_plot = False
if 'show_num_plot' not in st.session_state:
    st.session_state.show_num_plot = False
if 'cat_fig' not in st.session_state:
    st.session_state.cat_fig = None
if 'num_fig' not in st.session_state:
    st.session_state.num_fig = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None


def clear_all_cache():
    """Полностью очищает все кэши"""
    st.session_state.trained_models = {}
    st.session_state.predictions_cache = {}
    st.session_state.shap_cache = {}
    st.session_state.data_dict = None
    st.session_state.X1 = None
    st.session_state.y1 = None
    st.session_state.X2 = None
    st.session_state.y2 = None
    st.session_state.split_done = False
    st.session_state.show_cat_plot = False
    st.session_state.show_num_plot = False

    if st.session_state.cat_fig:
        plt.close(st.session_state.cat_fig)
        st.session_state.cat_fig = None
    if st.session_state.num_fig:
        plt.close(st.session_state.num_fig)
        st.session_state.num_fig = None

    return "Все кэши очищены"


def plot_categorical_comparison(df1, df2, categorical_columns):
    if len(categorical_columns) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет категориальных признаков", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    n_rows = len(categorical_columns)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(categorical_columns):
        first_counts = df1[col].value_counts()
        axes[i, 0].pie(first_counts.values, labels=first_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i, 0].set_title(f'{col} (первая выборка)\nn={len(df1)}')
        second_counts = df2[col].value_counts()
        axes[i, 1].pie(second_counts.values, labels=second_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i, 1].set_title(f'{col} (вторая выборка)\nn={len(df2)}')
    plt.tight_layout()
    return fig


def plot_numerical_comparison(df1, df2, numerical_columns):
    if len(numerical_columns) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет числовых признаков", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    n_rows = len(numerical_columns)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(numerical_columns):
        axes[i, 0].hist(df1[col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[i, 0].axvline(df1[col].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Среднее: {df1[col].mean():.3f}')
        axes[i, 0].set_title(f'{col} (первая выборка)\nn={len(df1)}')
        axes[i, 0].legend()
        axes[i, 1].hist(df2[col].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[i, 1].axvline(df2[col].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Среднее: {df2[col].mean():.3f}')
        axes[i, 1].set_title(f'{col} (вторая выборка)\nn={len(df2)}')
        axes[i, 1].legend()
    plt.tight_layout()
    return fig


def get_or_compute_shap(cluster, model_key, X_data, model):
    cache_key = f"{cluster}_{model_key}_shap"
    if cache_key in st.session_state.shap_cache:
        return st.session_state.shap_cache[cache_key]

    with st.spinner(f"Вычисление SHAP для {model_key} (всего {len(X_data)} объектов)..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    st.session_state.shap_cache[cache_key] = {
        'shap_values': shap_values,
        'X_data': X_data,
        'expected_value': explainer.expected_value,
        'feature_names': X_data.columns.tolist()
    }
    return st.session_state.shap_cache[cache_key]


def create_shap_summary_plot(shap_values, X_data, feature_names, plot_type='dot', title=None):
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type=plot_type, show=False, max_display=20)
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    return plt.gcf()


def create_shap_decision_plot(shap_values, expected_value, X_data, feature_names, title=None):
    plt.figure(figsize=(12, 7))
    base_value = expected_value if expected_value is not None else 0
    shap.decision_plot(base_value, shap_values, X_data.values, feature_names=feature_names, show=False)
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    return plt.gcf()


def create_shap_heatmap_plot(shap_values, X_data, feature_names, title=None):
    plt.figure(figsize=(14, 8))
    shap.plots.heatmap(shap.Explanation(values=shap_values, base_values=0, data=X_data.values,
                                        feature_names=feature_names), show=False)
    if title:
        plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    return plt.gcf()


def create_scatter_plot(y_true, y_pred, title):
    plt.figure(figsize=(8, 5))
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', edgecolors='white', s=50)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_histogram_plot(y_true, y_pred, title):
    plt.figure(figsize=(8, 5))
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    plt.hist(y_true, bins=30, alpha=0.5, color='blue', label='Реальные значения', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, color='red', label='Предсказанные значения', edgecolor='black')
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def get_datasets(dataset, column):
    if dataset is None:
        return None, "Сначала загрузите данные!"
    if column not in dataset.columns:
        return None, f"Колонка '{column}' не найдена!"
    unique_values = sorted(dataset[column].unique())
    datasets = [dataset[dataset[column] == value].copy().drop(columns=[column]) for value in unique_values]
    return datasets, f"Создано {len(datasets)} поднаборов"


def remove_singles(dataset, stratified_column):
    few = dataset[stratified_column].value_counts()
    categories = list(few[few == 1].index)
    singles = dataset[dataset[stratified_column].isin(categories)]
    dataset = dataset.drop(singles.index)
    return dataset, singles


def make_datasets(datasets, stratified_column='population', second_size=0.5, random_state=42):
    if not datasets:
        return None, None, "Нет данных для разбиения!"
    first_datasets, second_datasets = [], []
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
        if dataset[stratified_column].nunique() > 1 and int(dataset.shape[0] * second_size) >= dataset[
            stratified_column].nunique():
            stratify = dataset[stratified_column]
        first_size = 1.0 - second_size
        if int(dataset.shape[0] * first_size) == 0:
            if first_datasets:
                first_datasets[-1] = pd.concat([first_datasets[-1], dataset])
            else:
                first_datasets.append(dataset)
        else:
            local_first, local_second = train_test_split(dataset, test_size=second_size, stratify=stratify,
                                                         random_state=random_state)
            first_datasets.append(local_first)
            second_datasets.append(local_second)
    df1 = pd.concat(first_datasets) if first_datasets else pd.DataFrame()
    df2 = pd.concat(second_datasets) if second_datasets else pd.DataFrame()
    return df1, df2, f"Первая выборка: {len(df1)} строк, Вторая выборка: {len(df2)} строк"


def split_data(df, group_column, stratify_column, second_size, target_column, random_state):
    if df is None:
        return None, None, None, None, None, None, None, "Сначала загрузите данные!"

    clear_all_cache()

    df = df.copy()
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category')

    if target_column not in df.columns:
        return None, None, None, None, None, None, None, f"Целевая переменная '{target_column}' не найдена"
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return None, None, None, None, None, None, None, f"Целевая переменная '{target_column}' должна быть числовой"
    if group_column not in df.columns:
        return None, None, None, None, None, None, None, f"Колонка для группировки '{group_column}' не найдена"

    datasets, msg = get_datasets(df, group_column)
    if datasets is None:
        return None, None, None, None, None, None, None, msg

    df1, df2, split_msg = make_datasets(datasets, stratified_column=stratify_column, second_size=second_size,
                                        random_state=random_state)
    if df1 is None or len(df1) == 0:
        return None, None, None, None, None, None, None, split_msg

    X1, y1 = df1.drop(columns=[target_column]), df1[target_column]
    X2, y2 = df2.drop(columns=[target_column]), df2[target_column]
    data_dict = {'X1': X1, 'X2': X2, 'y1': y1, 'y2': y2, 'cluster_col': 'cluster'}

    return X1, y1, X2, y2, df1, df2, data_dict, f"Разбиение выполнено! Первая: {len(X1)}, Вторая: {len(X2)}"


def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def objective(trial, X_train, y_train, cat_features, cv, params_config):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', params_config['n_estimators_min'],
                                          params_config['n_estimators_max'], log=params_config['n_estimators_log']),
        'depth': trial.suggest_int('depth', params_config['depth_min'], params_config['depth_max']),
        'learning_rate': trial.suggest_float('learning_rate', params_config['learning_rate_min'],
                                             params_config['learning_rate_max'],
                                             log=params_config['learning_rate_log']),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', params_config['l2_leaf_reg_min'],
                                           params_config['l2_leaf_reg_max'], log=params_config['l2_leaf_reg_log']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', params_config['min_data_in_leaf_min'],
                                              params_config['min_data_in_leaf_max']),
        'verbose': False, 'random_seed': 42
    }
    model = CatBoostRegressor(**params)
    scores = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
        pred = model.predict(X_val)
        if params_config['scoring'] == 'r2':
            scores.append(r2_score(y_val, pred))
        elif params_config['scoring'] == 'neg_mean_squared_error':
            scores.append(-mean_squared_error(y_val, pred))
        else:
            scores.append(-mean_absolute_error(y_val, pred))
    return np.mean(scores)


def train_models_for_cluster(X1, y1, X2, y2, cluster, params_config, log_placeholder, progress_placeholder):
    if X1 is None or 'cluster' not in X1.columns:
        return "Сначала выполните разбиение!", None, None, 0, 0

    try:
        idx1 = X1[X1['cluster'].astype(str) == str(cluster)].index
        idx2 = X2[X2['cluster'].astype(str) == str(cluster)].index
    except Exception as e:
        return f"Ошибка фильтрации: {e}", None, None, 0, 0

    if len(idx1) == 0 or len(idx2) == 0:
        return f"Кластер {cluster} не найден", None, None, 0, 0

    X1_cluster = X1.loc[idx1].drop(columns=['cluster'])
    y1_cluster = y1.loc[idx1]
    X2_cluster = X2.loc[idx2].drop(columns=['cluster'])
    y2_cluster = y2.loc[idx2]
    cat_features = get_categorical_columns(X1_cluster)

    metric_display = {'r2': 'R²', 'neg_mean_squared_error': 'Среднеквадратичная ошибка',
                      'neg_mean_absolute_error': 'Средняя абсолютная ошибка'}[
        params_config['scoring']]
    results, models, best_params_list = [], [], []

    for model_num, (X_train, y_train, X_pred, y_pred, model_name, model_key) in enumerate([
        (X1_cluster, y1_cluster, X2_cluster, y2_cluster,
         "Модель 1 (обучена на первой выборке, предсказывает для второй)",
         "model1"),
        (X2_cluster, y2_cluster, X1_cluster, y1_cluster,
         "Модель 2 (обучена на второй выборке, предсказывает для первой)",
         "model2")
    ], 1):
        progress_placeholder.info(f"Обработка: {model_name}")
        study = create_study(direction='maximize', study_name=f'CatBoost_{cluster}_{model_key}')
        best_value, best_trial_number, best_params_str = -float('inf'), None, ""

        def callback(study, trial):
            nonlocal best_value, best_trial_number, best_params_str
            value = trial.value
            display_value = -value if params_config['scoring'] != 'r2' else value
            is_best = value > best_value
            if is_best:
                best_value = value
                best_trial_number = trial.number + 1
                best_params_str = f"глубина={trial.params.get('depth', 'Н/Д')}, темп={trial.params.get('learning_rate', 'Н/Д'):.4f}, деревьев={trial.params.get('n_estimators', 'Н/Д')}"
            current_best_display = -best_value if params_config['scoring'] != 'r2' else best_value
            log_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ                   ║
║                        {model_name[:50]}                         ║
╚══════════════════════════════════════════════════════════════════╝

ТЕКУЩАЯ ИТЕРАЦИЯ:
   Попытка: {trial.number + 1} / {params_config['n_trials']}
   {metric_display}: {display_value:.6f}
   {' НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ!' if is_best else ''}

ЛУЧШИЙ РЕЗУЛЬТАТ:
   Попытка #{best_trial_number if best_trial_number else 'Н/Д'}
   {metric_display}: {current_best_display:.6f}
   {best_params_str}

{'─' * 70}
"""
            log_placeholder.code(log_text, language="text")
            time.sleep(0.05)

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cat_features, params_config['cv'], params_config),
            n_trials=params_config['n_trials'], callbacks=[callback])

        best_params = study.best_params
        best_display = -study.best_value if params_config['scoring'] != 'r2' else study.best_value
        best_params_list.append(best_params)

        final_log = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОПТИМИЗАЦИЯ ЗАВЕРШЕНА                         ║
╚══════════════════════════════════════════════════════════════════╝

ЛУЧШИЕ ПАРАМЕТРЫ:
   Количество деревьев: {best_params.get('n_estimators', 'Н/Д')}
   Глубина деревьев: {best_params.get('depth', 'Н/Д')}
   Темп обучения: {best_params.get('learning_rate', 'Н/Д'):.4f}
   L2 регуляризация: {best_params.get('l2_leaf_reg', 'Н/Д'):.4f}
   Мин. данных в листе: {best_params.get('min_data_in_leaf', 'Н/Д')}

ЛУЧШЕЕ ЗНАЧЕНИЕ {metric_display} (CV): {best_display:.6f}
"""
        log_placeholder.code(final_log, language="text")

        progress_placeholder.info(f"Обучение финальной модели {model_num}/2...")
        final_model = CatBoostRegressor(**best_params, cat_features=cat_features, verbose=False, random_seed=42)
        final_model.fit(X_train, y_train, cat_features=cat_features)

        st.session_state.shap_cache[f"{cluster}_{model_key}_model"] = final_model
        st.session_state.shap_cache[f"{cluster}_{model_key}_X_pred"] = X_pred

        predictions = final_model.predict(X_pred)

        mse = mean_squared_error(y_pred, predictions)
        mae = mean_absolute_error(y_pred, predictions)
        r2 = r2_score(y_pred, predictions)

        result_text = f"""
Модель: {model_name}
{'─' * 70}
ЛУЧШИЕ ПАРАМЕТРЫ:
   Количество деревьев: {best_params.get('n_estimators', 'Н/Д')}
   Глубина деревьев: {best_params.get('depth', 'Н/Д')}
   Темп обучения: {best_params.get('learning_rate', 'Н/Д'):.4f}
   L2 регуляризация: {best_params.get('l2_leaf_reg', 'Н/Д'):.4f}
   Мин. данных в листе: {best_params.get('min_data_in_leaf', 'Н/Д')}

ЛУЧШЕЕ ЗНАЧЕНИЕ {metric_display} (CV): {best_display:.6f}

МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:
   Среднеквадратичная ошибка (MSE): {mse:.6f}
   Средняя абсолютная ошибка (MAE): {mae:.6f}
   Коэффициент детерминации (R²): {r2:.6f}
"""
        results.append(result_text)
        models.append(final_model)

    st.session_state.trained_models[str(cluster)] = {
        'model1': models[0],
        'model2': models[1],
        'params1': best_params_list[0] if len(best_params_list) > 0 else {},
        'params2': best_params_list[1] if len(best_params_list) > 1 else {},
        'X1_size': len(X1_cluster),
        'X2_size': len(X2_cluster)
    }
    preds1, preds2 = models[0].predict(X2_cluster), models[1].predict(X1_cluster)
    st.session_state.predictions_cache[str(cluster)] = {'preds1': preds1, 'preds2': preds2}
    return "\n\n".join(results), preds1[:10], preds2[:10], len(X1_cluster), len(X2_cluster)


def visualize_cluster(data_dict, cluster, plot_type, summary_plot_type="dot"):
    if data_dict is None:
        return None, None, "Нет данных"

    X1, X2, y1, y2 = data_dict['X1'], data_dict['X2'], data_dict['y1'], data_dict['y2']
    idx1 = X1[X1['cluster'].astype(str) == str(cluster)].index
    idx2 = X2[X2['cluster'].astype(str) == str(cluster)].index
    X1_cl, X2_cl = X1.loc[idx1].drop(columns=['cluster']), X2.loc[idx2].drop(columns=['cluster'])
    y1_cl, y2_cl = y1.loc[idx1].values.ravel(), y2.loc[idx2].values.ravel()

    cache = st.session_state.predictions_cache.get(str(cluster))
    if cache is None:
        return None, None, "Нет предсказаний для этого кластера"

    preds1, preds2 = cache['preds1'], cache['preds2']
    fig1, fig2 = None, None

    if plot_type == 'scatter':
        fig1 = create_scatter_plot(y2_cl, preds1, title=f"Модель 1: обучена на кластере {cluster}")
        fig2 = create_scatter_plot(y1_cl, preds2, title=f"Модель 2: обучена на кластере {cluster}")

    elif plot_type == 'histogram':
        fig1 = create_histogram_plot(y2_cl, preds1, title=f"Модель 1: обучена на кластере {cluster}")
        fig2 = create_histogram_plot(y1_cl, preds2, title=f"Модель 2: обучена на кластере {cluster}")

    elif plot_type == 'shap_summary':
        model1 = st.session_state.trained_models[str(cluster)]['model1']
        shap_data1 = get_or_compute_shap(str(cluster), "model1", X2_cl, model1)
        if shap_data1:
            fig1 = create_shap_summary_plot(
                shap_data1['shap_values'],
                shap_data1['X_data'],
                shap_data1['feature_names'],
                summary_plot_type,
                f"Модель 1\n(всего {len(X2_cl)} объектов)"
            )
        model2 = st.session_state.trained_models[str(cluster)]['model2']
        shap_data2 = get_or_compute_shap(str(cluster), "model2", X1_cl, model2)
        if shap_data2:
            fig2 = create_shap_summary_plot(
                shap_data2['shap_values'],
                shap_data2['X_data'],
                shap_data2['feature_names'],
                summary_plot_type,
                f"Модель 2\n(всего {len(X1_cl)} объектов)"
            )

    elif plot_type == 'shap_decision':
        model1 = st.session_state.trained_models[str(cluster)]['model1']
        shap_data1 = get_or_compute_shap(str(cluster), "model1", X2_cl, model1)
        if shap_data1:
            fig1 = create_shap_decision_plot(
                shap_data1['shap_values'],
                shap_data1['expected_value'],
                shap_data1['X_data'],
                shap_data1['feature_names'],
                f"Модель 1\n(всего {len(X2_cl)} объектов)"
            )
        model2 = st.session_state.trained_models[str(cluster)]['model2']
        shap_data2 = get_or_compute_shap(str(cluster), "model2", X1_cl, model2)
        if shap_data2:
            fig2 = create_shap_decision_plot(
                shap_data2['shap_values'],
                shap_data2['expected_value'],
                shap_data2['X_data'],
                shap_data2['feature_names'],
                f"Модель 2\n(всего {len(X1_cl)} объектов)"
            )

    elif plot_type == 'shap_heatmap':
        model1 = st.session_state.trained_models[str(cluster)]['model1']
        shap_data1 = get_or_compute_shap(str(cluster), "model1", X2_cl, model1)
        if shap_data1:
            fig1 = create_shap_heatmap_plot(
                shap_data1['shap_values'],
                shap_data1['X_data'],
                shap_data1['feature_names'],
                f"Модель 1\n(всего {len(X2_cl)} объектов)"
            )
        model2 = st.session_state.trained_models[str(cluster)]['model2']
        shap_data2 = get_or_compute_shap(str(cluster), "model2", X1_cl, model2)
        if shap_data2:
            fig2 = create_shap_heatmap_plot(
                shap_data2['shap_values'],
                shap_data2['X_data'],
                shap_data2['feature_names'],
                f"Модель 2\n(всего {len(X1_cl)} объектов)"
            )

    return fig1, fig2, f"Графики построены для кластера {cluster}"


st.set_page_config(page_title="Предсказание уровня успеваемости", layout="wide")

st.title("Предсказание уровня успеваемости")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Загрузка данных",
    "Обучение моделей",
    "Визуализация",
    "Сохранённые модели",
    "Выгрузка результатов"
])

# Вкладка 1: Загрузка данных
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("CSV файл", type=["csv"], key="file_uploader_main")

        if uploaded_file is not None:
            if st.session_state.current_file_name != uploaded_file.name:
                clear_all_cache()
                st.session_state.current_file_name = uploaded_file.name
                st.rerun()

            df = pd.read_csv(uploaded_file, index_col=0)
            st.success(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
            st.dataframe(df.head(), use_container_width=True)

            all_cols = df.columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'cluster' in df.columns and 'cluster' not in cat_cols:
                cat_cols.append('cluster')
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            group_column = st.selectbox("Колонка для группировки", all_cols, key="group_column")
            stratify_column = st.selectbox("Колонка для стратификации", cat_cols if cat_cols else all_cols,
                                           key="stratify_column")
            second_size = st.slider("Доля второй выборки", 0.1, 0.5, 0.3, 0.05, key="second_size")
            target_column = st.selectbox("Целевая переменная", num_cols, key="target_column")
            random_state = st.number_input("Случайное зерно (random_state)", 1, 999, 42, step=1,
                                           key="random_state_split")

            if st.button("Выполнить разбиение", type="primary", key="split_button"):
                with st.spinner("Разбиение..."):
                    res = split_data(df, group_column, stratify_column, second_size, target_column, random_state)
                    st.session_state.X1, st.session_state.y1, st.session_state.X2, st.session_state.y2, \
                        st.session_state.df1, st.session_state.df2, st.session_state.data_dict, msg = res

                    if st.session_state.X1 is not None:
                        st.session_state.split_done = True
                        st.success(msg)
                        st.write(f"**Первая выборка:** {len(st.session_state.X1)} строк")
                        st.write(f"**Вторая выборка:** {len(st.session_state.X2)} строк")
                    else:
                        st.error(msg)

    with col2:
        if st.session_state.split_done and st.session_state.df1 is not None:
            st.markdown("### Визуализация распределений")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Категориальные признаки", key="cat_btn"):
                    if not st.session_state.show_cat_plot:
                        cat_cols_viz = st.session_state.df1.select_dtypes(
                            include=['object', 'category']).columns.tolist()
                        if cat_cols_viz:
                            st.session_state.cat_fig = plot_categorical_comparison(st.session_state.df1,
                                                                                   st.session_state.df2, cat_cols_viz)
                            st.session_state.show_cat_plot = True
                        else:
                            st.info("Нет категориальных признаков")
                    else:
                        st.session_state.show_cat_plot = False
                        if st.session_state.cat_fig:
                            plt.close(st.session_state.cat_fig)
                            st.session_state.cat_fig = None

            with col_btn2:
                if st.button("Числовые признаки", key="num_btn"):
                    if not st.session_state.show_num_plot:
                        num_cols_viz = st.session_state.df1.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols_viz:
                            st.session_state.num_fig = plot_numerical_comparison(st.session_state.df1,
                                                                                 st.session_state.df2, num_cols_viz)
                            st.session_state.show_num_plot = True
                        else:
                            st.info("Нет числовых признаков")
                    else:
                        st.session_state.show_num_plot = False
                        if st.session_state.num_fig:
                            plt.close(st.session_state.num_fig)
                            st.session_state.num_fig = None

            st.markdown("---")
            if st.session_state.show_cat_plot and st.session_state.cat_fig:
                st.markdown("#### Категориальные признаки")
                st.pyplot(st.session_state.cat_fig)
                st.markdown("---")
            if st.session_state.show_num_plot and st.session_state.num_fig:
                st.markdown("#### Числовые признаки")
                st.pyplot(st.session_state.num_fig)

# Вкладка 2: Обучение моделей
with tab2:
    if not st.session_state.split_done:
        st.warning("Сначала выполните разбиение данных на вкладке 1")
    else:
        if 'cluster' not in st.session_state.X1.columns:
            st.error("В данных нет колонки 'cluster'")
        else:
            clusters = sorted(st.session_state.X1['cluster'].unique())

            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.markdown("### Выбор кластера")
                cluster = st.selectbox("Кластер для обучения", clusters, key="train_cluster")

                st.markdown("### Параметры оптимизации")
                n_trials = st.slider("Количество попыток", 5, 100, 20, 5, key="n_trials")
                cv = st.slider("Количество фолдов (CV)", 2, 10, 3, 1, key="cv_folds")
                scoring = st.selectbox("Метрика для оптимизации",
                                       ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
                                       format_func=lambda x:
                                       {"r2": "R² (максимизация)",
                                        "neg_mean_squared_error": "MSE (минимизация)",
                                        "neg_mean_absolute_error": "MAE (минимизация)"}[x], key="scoring_metric")

                st.markdown("### Границы гиперпараметров")

                with st.expander("Количество деревьев (n_estimators)"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: n_est_min = st.number_input("Минимум", 10, 500, 100, step=1, key="n_est_min")
                    with col_b: n_est_max = st.number_input("Максимум", 50, 2000, 1000, step=1, key="n_est_max")
                    with col_c: n_est_log = st.checkbox("Логарифмический масштаб", True, key="n_est_log")

                with st.expander("Глубина деревьев (depth)"):
                    col_a, col_b = st.columns(2)
                    with col_a: depth_min = st.number_input("Минимум", 2, 6, 4, step=1, key="depth_min")
                    with col_b: depth_max = st.number_input("Максимум", 3, 16, 10, step=1, key="depth_max")

                with st.expander("L2 регуляризация (l2_leaf_reg)"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: l2_min = st.number_input("Минимум", 0.001, 0.1, 0.01, step=0.01, format="%.4f",
                                                         key="l2_min")
                    with col_b: l2_max = st.number_input("Максимум", 0.5, 10.0, 1.0, step=0.5, key="l2_max")
                    with col_c: l2_log = st.checkbox("Логарифмический масштаб", True, key="l2_log")

                with st.expander("Темп обучения (learning_rate)"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: lr_min = st.number_input("Минимум", 0.001, 0.1, 0.01, step=0.005, format="%.4f",
                                                         key="lr_min")
                    with col_b: lr_max = st.number_input("Максимум", 0.05, 1.0, 0.3, step=0.05, key="lr_max")
                    with col_c: lr_log = st.checkbox("Логарифмический масштаб", True, key="lr_log")

                with st.expander("Мин. данных в листе (min_data_in_leaf)"):
                    col_a, col_b = st.columns(2)
                    with col_a: mleaf_min = st.number_input("Минимум", 1, 20, 2, step=1, key="mleaf_min")
                    with col_b: mleaf_max = st.number_input("Максимум", 2, 100, 50, step=1, key="mleaf_max")

                train_btn = st.button("Запустить оптимизацию", type="primary", use_container_width=True,
                                      key="train_btn")

            with col2:
                st.markdown("### Логи оптимизации (обновляются в реальном времени)")
                log_container = st.empty()
                progress_container = st.empty()
                result_container = st.empty()

                if train_btn:
                    params_config = {
                        'n_trials': n_trials, 'cv': cv, 'scoring': scoring,
                        'n_estimators_min': int(n_est_min), 'n_estimators_max': int(n_est_max),
                        'n_estimators_log': n_est_log,
                        'depth_min': int(depth_min), 'depth_max': int(depth_max),
                        'learning_rate_min': float(lr_min), 'learning_rate_max': float(lr_max),
                        'learning_rate_log': lr_log,
                        'l2_leaf_reg_min': float(l2_min), 'l2_leaf_reg_max': float(l2_max), 'l2_leaf_reg_log': l2_log,
                        'min_data_in_leaf_min': int(mleaf_min), 'min_data_in_leaf_max': int(mleaf_max),
                    }

                    with st.spinner(f"Оптимизация для кластера {cluster}..."):
                        result, preds1, preds2, sz1, sz2 = train_models_for_cluster(
                            st.session_state.X1, st.session_state.y1, st.session_state.X2, st.session_state.y2,
                            cluster, params_config, log_container, progress_container
                        )

                        result_container.text_area("Результаты оптимизации", result, height=250, key="result_area")
                        progress_container.success(
                            f"Обучение завершено! Первая выборка: {sz1} строк, Вторая: {sz2} строк")

                        if preds1 is not None:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**Предсказания модели 1 (первые 10):**")
                                st.write(preds1)
                            with col_b:
                                st.write("**Предсказания модели 2 (первые 10):**")
                                st.write(preds2)

# Вкладка 3: Визуализация
with tab3:
    if not st.session_state.trained_models:
        st.info("Сначала обучите модели на вкладке 2")
    else:
        st.markdown("## Визуализация моделей")

        clusters_viz = sorted(st.session_state.trained_models.keys())
        cluster_viz = st.selectbox("Выберите кластер", clusters_viz, key="viz_cluster")

        plot_type = st.radio(
            "Тип визуализации",
            ["scatter", "histogram", "shap_summary", "shap_decision", "shap_heatmap"],
            format_func=lambda x: {
                "scatter": "Точечная диаграмма (реальные против предсказанных)",
                "histogram": "Гистограммы распределений",
                "shap_summary": "SHAP Summary (важность признаков)",
                "shap_decision": "SHAP Decision (влияние на предсказание)",
                "shap_heatmap": "SHAP Heatmap (тепловая карта)",
            }[x],
            horizontal=False,
            key="plot_type_radio"
        )

        summary_plot_type = "dot"
        if plot_type == "shap_summary":
            st.markdown("#### Настройки SHAP графика")
            summary_plot_type = st.radio(
                "Тип Summary графика",
                ["dot", "bar"],
                format_func=lambda x: {
                    "dot": "Точечный график (распределение влияния признаков)",
                    "bar": "Столбчатая диаграмма (среднее абсолютное влияние)",
                }[x],
                horizontal=True,
                key="summary_type_radio"
            )

        if st.button("Построить графики", key="build_plots_btn"):
            with st.spinner("Строим графики..."):
                fig1, fig2, msg = visualize_cluster(
                    st.session_state.data_dict,
                    cluster_viz,
                    plot_type,
                    summary_plot_type
                )

                if fig1 is not None or fig2 is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        if fig1 is not None:
                            st.pyplot(fig1)
                            plt.close()
                        else:
                            st.info("График для модели 1 не создан")
                    with col2:
                        if fig2 is not None:
                            st.pyplot(fig2)
                            plt.close()
                        else:
                            st.info("График для модели 2 не создан")
                    st.success(msg)
                else:
                    st.error(msg)

# Вкладка 4: Сохранённые модели
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Показать все модели", key="show_models_btn"):
            if not st.session_state.trained_models:
                st.info("Кэш пуст")
            else:
                for cluster, models_data in st.session_state.trained_models.items():
                    with st.expander(f"Кластер {cluster}"):
                        st.write(f"**Размер первой выборки:** {models_data.get('X1_size', 'Н/Д')}")
                        st.write(f"**Размер второй выборки:** {models_data.get('X2_size', 'Н/Д')}")

                        st.markdown("**Модель 1 (первая выборка → вторая выборка):**")
                        if models_data.get('model1') is not None:
                            params1 = models_data.get('params1', {})
                            if params1:
                                st.write("   **Гиперпараметры:**")
                                st.write(
                                    f"      - Количество деревьев (n_estimators): {params1.get('n_estimators', 'Н/Д')}")
                                st.write(f"      - Глубина деревьев (depth): {params1.get('depth', 'Н/Д')}")
                                st.write(
                                    f"      - Темп обучения (learning_rate): {params1.get('learning_rate', 'Н/Д'):.4f}")
                                st.write(
                                    f"      - L2 регуляризация (l2_leaf_reg): {params1.get('l2_leaf_reg', 'Н/Д'):.4f}")
                                st.write(
                                    f"      - Мин. данных в листе (min_data_in_leaf): {params1.get('min_data_in_leaf', 'Н/Д')}")
                            else:
                                st.write("   Модель обучена (параметры не сохранены)")
                        else:
                            st.write("   Модель не обучена")

                        st.markdown("**Модель 2 (вторая выборка → первая выборка):**")
                        if models_data.get('model2') is not None:
                            params2 = models_data.get('params2', {})
                            if params2:
                                st.write("   **Гиперпараметры:**")
                                st.write(
                                    f"      - Количество деревьев (n_estimators): {params2.get('n_estimators', 'Н/Д')}")
                                st.write(f"      - Глубина деревьев (depth): {params2.get('depth', 'Н/Д')}")
                                st.write(
                                    f"      - Темп обучения (learning_rate): {params2.get('learning_rate', 'Н/Д'):.4f}")
                                st.write(
                                    f"      - L2 регуляризация (l2_leaf_reg): {params2.get('l2_leaf_reg', 'Н/Д'):.4f}")
                                st.write(
                                    f"      - Мин. данных в листе (min_data_in_leaf): {params2.get('min_data_in_leaf', 'Н/Д')}")
                            else:
                                st.write("   Модель обучена (параметры не сохранены)")
                        else:
                            st.write("   Модель не обучена")

        if st.button("Показать SHAP кэш", key="show_shap_cache_btn"):
            shap_keys = [k for k in st.session_state.shap_cache.keys() if k.endswith('_shap')]
            if not shap_keys:
                st.info("SHAP кэш пуст")
            else:
                for key in shap_keys:
                    with st.expander(f"SHAP: {key}"):
                        shap_data = st.session_state.shap_cache[key]
                        st.write(f"**Размер данных:** {len(shap_data['X_data'])} объектов")
                        st.write(f"**Количество признаков:** {len(shap_data['feature_names'])}")
                        st.write(f"**Базовое значение:** {shap_data['expected_value']:.4f}")

    with col2:
        if st.button("Очистить кэш моделей", type="secondary", key="clear_models_btn"):
            st.session_state.trained_models = {}
            st.session_state.predictions_cache = {}
            st.success("Кэш моделей очищен")

        if st.button("Очистить SHAP кэш", type="secondary", key="clear_shap_btn"):
            shap_keys = [k for k in st.session_state.shap_cache.keys() if k.endswith('_shap')]
            for key in shap_keys:
                del st.session_state.shap_cache[key]
            st.success(f"SHAP кэш очищен (удалено {len(shap_keys)} записей)")

# Вкладка 5: Выгрузка результатов
with tab5:
    if not st.session_state.trained_models:
        st.warning("Нет обученных моделей")
    else:
        export_cluster = st.selectbox("Кластер для выгрузки", sorted(st.session_state.trained_models.keys()),
                                      key="export_cluster")
        filename = st.text_input("Имя файла", f"cluster_{export_cluster}", key="filename")
        pred_col_name = st.text_input("Название колонки с предсказаниями", "predictions", key="pred_col")

        if st.button("Скачать CSV", key="download_csv_btn"):
            data = st.session_state.data_dict
            cache = st.session_state.predictions_cache.get(str(export_cluster))
            if cache and data:
                X1, X2 = data['X1'], data['X2']
                y1, y2 = data['y1'], data['y2']

                idx1 = X1[X1['cluster'].astype(str) == str(export_cluster)].index
                idx2 = X2[X2['cluster'].astype(str) == str(export_cluster)].index

                df_model1 = X2.loc[idx2].copy()
                df_model1[y2.name] = y2.loc[idx2].values.ravel()
                df_model1[pred_col_name] = cache['preds1']
                df_model1['residuals'] = y2.loc[idx2].values.ravel() - cache['preds1']

                df_model2 = X1.loc[idx1].copy()
                df_model2[y1.name] = y1.loc[idx1].values.ravel()
                df_model2[pred_col_name] = cache['preds2']
                df_model2['residuals'] = y1.loc[idx1].values.ravel() - cache['preds2']

                result_df = pd.concat([df_model1, df_model2])
                temp_path = os.path.join(tempfile.gettempdir(), f"{filename}.csv")
                result_df.to_csv(temp_path, index=True, encoding='utf-8-sig')

                with open(temp_path, 'rb') as f:
                    st.download_button("Скачать файл", f, file_name=f"{filename}.csv", key="download_file_btn")

                st.success(f"Выгружено {len(result_df)} строк")
            else:
                st.error("Данные не найдены")