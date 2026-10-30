"""
Обучение и сравнение моделей машинного обучения для прогнозирования дефолта автокредитов.
Включает масштабирование данных, балансировку классов (SMOTE) и сохранение всех артефактов.
"""

# 1. РЕШЕНИЕ ПРОБЛЕМЫ OPENBLAS: Ограничиваем количество потоков ДО импорта numpy/sklearn
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.data_loader import load_or_generate_data
from src.preprocessing import prepare_data, apply_balancing, FEATURES


def train_all_models():
    """Обучает все модели, сравнивает их и сохраняет лучшую + scaler."""
    print("🔄 Загрузка данных...")
    df = load_or_generate_data()
    
    print("🔄 Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 2. МАСШТАБИРОВАНИЕ ДАННЫХ (Критически важно для Logistic Regression и расстояний в SMOTE)
    print("🔄 Масштабирование признаков (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. БАЛАНСИРОВКА КЛАССОВ (Применяем ТОЛЬКО к обучающей выборке!)
    print("🔄 Применение SMOTE для балансировки обучающей выборки...")
    X_train_bal, y_train_bal = apply_balancing(X_train_scaled, y_train, strategy='smote', ratio=0.3)
    
    print(f"   Доля дефолтов до SMOTE: {y_train.mean():.2%}")
    print(f"   Доля дефолтов после SMOTE: {y_train_bal.mean():.2%}")

    # Расчёт веса для алгоритмической балансировки (для XGBoost и CatBoost)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # 4. ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ
    models = {
        # max_iter увеличен для гарантии сходимости, но со StandardScaler обычно хватает 100-200
        'LogReg_scaled': LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
        
        'RandomForest_balanced': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1
        ),
        
        'XGBoost_scaled': XGBClassifier(
            scale_pos_weight=pos_weight, n_estimators=200, max_depth=6, 
            learning_rate=0.1, eval_metric='aucpr', random_state=42, verbosity=0, n_jobs=-1
        ),
        
        'CatBoost_scaled': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1, scale_pos_weight=pos_weight,
            verbose=0, random_state=42
        ),
    }

    results = []
    os.makedirs('models', exist_ok=True)

    print("\n🚀 Начало обучения моделей...\n" + "-"*50)
    
    for name, model in models.items():
        print(f"Обучение: {name}...")
        
        # Обучаем на масштабированных и сбалансированных данных
        model.fit(X_train_bal, y_train_bal)
        
        # Предсказываем на МАСШТАБИРОВАННЫХ, но НЕСБАЛАНСИРОВАННЫХ тестовых данных 
        # (чтобы оценить реальную работу в боевых условиях)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Расчёт метрик
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Бизнес-логика: стоимость ложного отказа (FP) = 5 000 руб., стоимость дефолта (FN) = 350 000 руб.
        business_loss = (fp * 5000) + (fn * 350000)

        results.append({
            'model': name,
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'PR-AUC': average_precision_score(y_test, y_proba), # Главная метрика при дисбалансе!
            'Business_Loss': business_loss,
        })

        # Сохраняем каждую обученную модель
        joblib.dump(model, f'models/{name}.pkl')
        print(f"   ✅ {name}: PR-AUC={results[-1]['PR-AUC']:.3f}, Recall={results[-1]['Recall']:.3f}")

    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ И АРТЕФАКТОВ
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/comparison.csv', index=False)
    print("\n📊 Таблица сравнения сохранена в models/comparison.csv")

    # Находим лучшую модель по ключевой метрике PR-AUC
    best = results_df.loc[results_df['PR-AUC'].idxmax()]
    
    # Сохраняем информацию о лучшей модели
    joblib.dump({
        'model_name': best['model'], 
        'features': FEATURES
    }, 'models/best_model_info.pkl')
    
    # 6. КРИТИЧЕСКИ ВАЖНО: Сохраняем обученный scaler для инференса!
    joblib.dump(scaler, 'models/scaler.pkl')
    print("💾 Масштабатор (scaler.pkl) успешно сохранён.")
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best['model']}")
    print(f"   PR-AUC: {best['PR-AUC']:.3f}")
    print(f"   Ожидаемый бизнес-убыток на тесте: {best['Business_Loss']:,.0f} руб.")
    
    return results_df


if __name__ == '__main__':
    train_all_models()