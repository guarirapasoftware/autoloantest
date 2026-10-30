#Генерация синтетических данных автокредитов
import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=12500, default_rate=0.068, random_state=42):
    #Генерирует реалистичные данные автокредитов
    np.random.seed(random_state)

    # Характеристики заёмщика
    age_raw = np.random.normal(loc=38, scale=10, size=n_samples)
    age = np.clip(age_raw, 18, 70).astype(int)
    income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples).astype(int)
    monthly_debt = (np.random.beta(2, 5, n_samples) * income * 0.6).astype(int)
    credit_history_score = np.random.randint(300, 850, n_samples)

    # Характеристики сделки
    loan_amount = np.random.randint(200000, 3000000, n_samples)
    term_months = np.random.choice([12, 24, 36, 48, 60, 72], n_samples)
    interest_rate = np.random.uniform(8, 22, n_samples).round(2)
    down_payment = (np.random.beta(2, 3, n_samples) * loan_amount * 0.3).astype(int)

    # Характеристики авто
    car_value = (loan_amount / np.random.uniform(0.7, 1.1, n_samples)).astype(int)
    car_year = np.random.randint(2005, 2025, n_samples)
    mileage = np.random.randint(5000, 250000, n_samples)

    # Расчёт производных признаков
    LTV = loan_amount / car_value * 100
    DTI = monthly_debt / income * 100
    car_age = 2026 - car_year
    down_payment_pct = down_payment / car_value * 100

    # Генерация целевой переменной с учётом факторов риска
    risk_score = (
        0.35 * (LTV > 100) +
        0.25 * (DTI > 50) +
        0.15 * (car_age > 10) +
        0.15 * (credit_history_score < 600) +
        0.10 * (down_payment_pct < 10)
    )
    noise = np.random.normal(0, 0.15, n_samples)
    prob_default = 1 / (1 + np.exp(-(risk_score + noise - 0.3) * 5))

    # Корректируем под нужную долю дефолтов
    threshold = np.percentile(prob_default, (1 - default_rate) * 100)
    default = (prob_default >= threshold).astype(int)

    df = pd.DataFrame({
        'loan_id': range(1, n_samples + 1),
        'age': age,
        'income': income,
        'monthly_debt': monthly_debt,
        'credit_history_score': credit_history_score,
        'loan_amount': loan_amount,
        'term_months': term_months,
        'interest_rate': interest_rate,
        'down_payment': down_payment,
        'car_value': car_value,
        'car_year': car_year,
        'mileage': mileage,
        'LTV': LTV.round(2),
        'DTI': DTI.round(2),
        'car_age': car_age,
        'down_payment_pct': down_payment_pct.round(2),
        'default': default,
    })

    return df

def load_or_generate_data(path='data/auto_loan_data.csv'):
    #Загружает данные или генерирует синтетические
    if os.path.exists(path):
        return pd.read_csv(path)
    os.makedirs('data', exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(path, index=False)
    print(f"✅ Сгенерирован датасет: {len(df)} записей, доля дефолтов: {df['default'].mean():.2%}")
    return df

if __name__ == '__main__':
    df = load_or_generate_data()
    print(df.head())
    print(f"\nРаспределение классов:\n{df['default'].value_counts()}")