#Предобработка данных и методы балансировки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

FEATURES = ['age', 'income', 'monthly_debt', 'credit_history_score',
            'loan_amount', 'term_months', 'interest_rate', 'down_payment',
            'car_value', 'car_year', 'mileage', 'LTV', 'DTI',
            'car_age', 'down_payment_pct']

def prepare_data(df, test_size=0.25, random_state=42):
    #Разделение на признаки и целевую переменную
    X = df[FEATURES]
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def apply_balancing(X_train, y_train, strategy='smote', ratio=0.3, random_state=42):
    #Применяет метод балансировки
    if strategy == 'none':
        return X_train, y_train
    elif strategy == 'smote':
        sampler = SMOTE(sampling_strategy=ratio, random_state=random_state)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy=ratio, random_state=random_state)
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res