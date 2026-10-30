"""
Инференс (предсказание) модели с логированием и правильным масштабированием данных.
Критически важно: входные данные должны быть масштабированы тем же StandardScaler, 
что и данные при обучении.
"""
import joblib
import json
import logging
import os
from datetime import datetime
import pandas as pd

# Создаем папку для логов, если её нет
os.makedirs('logs', exist_ok=True)

# Настраиваем логирование в JSON-формате для каждого запроса
logging.basicConfig(
    filename='logs/inference.log',
    level=logging.INFO,
    format='%(message)s'
)

class ScoringEngine:
    def __init__(self, model_name='CatBoost_scaled'):
        """Инициализация движка скоринга: загрузка модели, скалера и списка признаков."""
        model_path = f'models/{model_name}.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите src.train")
        
        # 1. Загружаем обученную модель
        self.model = joblib.load(model_path)
        self.model_name = model_name
        
        # 2. КРИТИЧЕСКИ ВАЖНО: Загружаем обученный масштабатор (StandardScaler)
        scaler_path = 'models/scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Файл scaler.pkl не найден. Перезапустите обучение (src.train).")
        self.scaler = joblib.load(scaler_path)
        
        # 3. Загружаем точный порядок признаков, на которых училась модель
        info = joblib.load('models/best_model_info.pkl')
        self.features = info['features']

    def predict(self, features: dict) -> dict:
        """
        Принимает словарь с параметрами заявки, возвращает вероятность дефолта и класс риска.
        """
        # 1. Создаем DataFrame строго в том порядке колонок, на котором училась модель
        df = pd.DataFrame([features])[self.features]
        
        # 2. МАСШТАБИРУЕМ входные данные тем же скалером, что и при обучении
        df_scaled = self.scaler.transform(df)
        
        # 3. Делаем предсказание на масштабированных данных
        # predict_proba возвращает массив [[вероятность_0, вероятность_1]], берем [0, 1]
        proba = float(self.model.predict_proba(df_scaled)[0, 1])

        # 4. Определяем класс риска на основе пороговых значений
        if proba > 0.50:
            risk = 'high'
        elif proba > 0.20:
            risk = 'medium'
        else:
            risk = 'low'

        # 5. Извлекаем топ-5 факторов риска (важность признаков) для интерпретации
        top_factors = []
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Сортируем индексы по убыванию важности и берем топ-5
            top_idx = importances.argsort()[-5:][::-1]
            top_factors = [(self.features[i], float(importances[i])) for i in top_idx]

        # 6. Логируем запрос и результат для MLOps-мониторинга
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model_version': self.model_name,
            'input_features': features,
            'prediction': round(proba, 4),
            'risk_class': risk,
        }
        logging.info(json.dumps(log_entry, ensure_ascii=False))

        return {
            'probability': proba,
            'risk_class': risk,
            'top_factors': top_factors,
        }


class ModelComparator:
    """Класс для сравнения всех обученных моделей."""
    
    def __init__(self):
        import os
        if not os.path.exists('models/comparison.csv'):
            raise FileNotFoundError("Файл comparison.csv не найден. Сначала запустите src.train")
        self.comparison_df = pd.read_csv('models/comparison.csv')
        self.scaler = joblib.load('models/scaler.pkl')
        info = joblib.load('models/best_model_info.pkl')
        self.features = info['features']
    
    def get_comparison_table(self):
        """Возвращает таблицу сравнения всех моделей."""
        return self.comparison_df
    
    def get_best_model_name(self):
        """Возвращает имя лучшей модели по PR-AUC."""
        return self.comparison_df.loc[self.comparison_df['PR-AUC'].idxmax(), 'model']
    
    def get_model_metrics(self, model_name):
        """Возвращает метрики конкретной модели."""
        row = self.comparison_df[self.comparison_df['model'] == model_name]
        return row.iloc[0].to_dict() if not row.empty else None
    
    def get_roc_curve_data(self, model_name, X_test, y_test):
        """Строит ROC-кривую для модели."""
        from sklearn.metrics import roc_curve, auc
        model = joblib.load(f'models/{model_name}.pkl')
        X_scaled = self.scaler.transform(X_test)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def get_pr_curve_data(self, model_name, X_test, y_test):
        """Строит PR-кривую для модели."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        model = joblib.load(f'models/{model_name}.pkl')
        X_scaled = self.scaler.transform(X_test)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        return precision, recall, ap
    
    def get_feature_importance(self, model_name):
        """Возвращает важность признаков для модели."""
        try:
            model = joblib.load(f'models/{model_name}.pkl')
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели {model_name}: {e}")
            return None
        
        # Для деревьев и бустингов
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return pd.DataFrame({
                'feature': self.features,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # Для линейных моделей (LogisticRegression)
        elif hasattr(model, 'coef_'):
            try:
                importances = abs(model.coef_[0])
                return pd.DataFrame({
                    'feature': self.features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            except Exception:
                return None
        
        return None

if __name__ == '__main__':
    # Тестовый запуск модуля для проверки работоспособности
    print("🔄 Инициализация ScoringEngine...")
    try:
        engine = ScoringEngine()
        
        # Тестовый кейс 1: Хороший заёмщик
        good_client = {
            'age': 40, 'income': 150000, 'monthly_debt': 10000,
            'credit_history_score': 750, 'loan_amount': 800000,
            'term_months': 36, 'interest_rate': 12.0, 'down_payment': 300000,
            'car_value': 1100000, 'car_year': 2022, 'mileage': 15000,
            'LTV': 72.73, 'DTI': 6.67, 'car_age': 4, 'down_payment_pct': 27.27,
        }
        res_good = engine.predict(good_client)
        print(f"\n✅ Хороший клиент: Вероятность дефолта = {res_good['probability']:.1%}, Риск = {res_good['risk_class'].upper()}")

        # Тестовый кейс 2: Рискованный заёмщик
        bad_client = {
            'age': 19, 'income': 35000, 'monthly_debt': 20000,
            'credit_history_score': 450, 'loan_amount': 1200000,
            'term_months': 84, 'interest_rate': 22.0, 'down_payment': 0,
            'car_value': 1000000, 'car_year': 2010, 'mileage': 180000,
            'LTV': 120.0, 'DTI': 57.14, 'car_age': 16, 'down_payment_pct': 0.0,
        }
        res_bad = engine.predict(bad_client)
        print(f"⚠️ Рискованный клиент: Вероятность дефолта = {res_bad['probability']:.1%}, Риск = {res_bad['risk_class'].upper()}")
        
        print("\n🎉 Инференс работает корректно! Вероятности реагируют на входные данные.")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании инференса: {e}")
        print("💡 Убедитесь, что вы запустили 'python -m src.train' и в папке models есть файлы .pkl и scaler.pkl")
