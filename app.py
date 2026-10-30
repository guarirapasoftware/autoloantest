"""
Веб-интерфейс системы скоринга автокредитов с дашбордом сравнения моделей.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

st.set_page_config(
    page_title="Скоринг автокредитов", 
    layout="wide", 
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# Загрузка движков
@st.cache_resource
def load_engine():
    from src.inference import ScoringEngine
    return ScoringEngine()

@st.cache_resource
def load_comparator():
    from src.inference import ModelComparator
    return ModelComparator()

@st.cache_resource
def load_test_data():
    """Загружает тестовые данные для построения кривых."""
    from src.data_loader import load_or_generate_data
    from src.preprocessing import prepare_data
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    return X_test, y_test

# Заголовок
st.title("🚗 Система прогнозирования дефолта автокредита")
st.markdown("*Курсовая работа: ML на несбалансированных данных*")

# Навигация через вкладки
tab1, tab2, tab3 = st.tabs(["🎯 Скоринг заявки", "📊 Сравнение моделей", "ℹ️ О системе"])

# ============================================================
# ВКЛАДКА 1: СКОРИНГ ЗАЯВКИ (основной функционал)
# ============================================================
with tab1:
    st.header("Оценка новой заявки на автокредит")
    
    engine = load_engine()
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("📋 Параметры заёмщика")
        age = st.number_input("Возраст", 18, 70, 35)
        income = st.number_input("Доход, руб./мес.", 20000, 2000000, 120000)
        monthly_debt = st.number_input("Платежи по другим кредитам, руб.", 0, 500000, 15000)
        credit_score = st.slider("Кредитный рейтинг", 300, 850, 700)
        
        st.subheader("💰 Параметры сделки")
        loan_amount = st.number_input("Сумма кредита, руб.", 100000, 10000000, 1000000)
        term = st.selectbox("Срок, мес.", [12, 24, 36, 48, 60, 72], index=4)
        rate = st.slider("Ставка, %", 8.0, 25.0, 14.5)
        down_payment = st.number_input("Первоначальный взнос, руб.", 0, 5000000, 200000)
        
        st.subheader("🚙 Параметры авто")
        car_value = st.number_input("Стоимость авто, руб.", 200000, 10000000, 1200000)
        car_year = st.slider("Год выпуска", 2005, 2025, 2020)
        mileage = st.number_input("Пробег, км", 0, 300000, 50000)
    
    # Расчёт производных признаков
    LTV = loan_amount / car_value * 100 if car_value > 0 else 0
    DTI = monthly_debt / income * 100 if income > 0 else 0
    car_age = 2026 - car_year
    down_payment_pct = down_payment / car_value * 100 if car_value > 0 else 0
    
    with col_right:
        st.subheader("📊 Производные показатели")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LTV", f"{LTV:.1f}%", 
                  delta="⚠️ высокий" if LTV > 100 else "норма",
                  delta_color="inverse" if LTV > 100 else "normal")
        c2.metric("DTI", f"{DTI:.1f}%", 
                  delta="⚠️ высокий" if DTI > 50 else "норма",
                  delta_color="inverse" if DTI > 50 else "normal")
        c3.metric("Возраст авто", f"{car_age} лет")
        c4.metric("Перв. взнос", f"{down_payment_pct:.1f}%")
        
        if st.button("🎯 Оценить риск", type="primary", use_container_width=True):
            # Жесткие бизнес-правила
            hard_reject = False
            reject_reason = ""
            
            if LTV > 150.0:
                hard_reject = True
                reject_reason = "LTV превышает 150%"
            elif DTI > 70.0:
                hard_reject = True
                reject_reason = "DTI превышает 70%"
            elif age < 21 and down_payment_pct < 20.0:
                hard_reject = True
                reject_reason = "Возраст до 21 года без существенного взноса"
            
            st.markdown("---")
            
            if hard_reject:
                st.error(f"🛑 **АВТОМАТИЧЕСКИЙ ОТКАЗ**\n\nПричина: {reject_reason}\n\n*Заявка не передана в ML-модель из-за нарушения политик риск-менеджмента.*")
            else:
                features = {
                    'age': age, 'income': income, 'monthly_debt': monthly_debt,
                    'credit_history_score': credit_score, 'loan_amount': loan_amount,
                    'term_months': term, 'interest_rate': rate, 'down_payment': down_payment,
                    'car_value': car_value, 'car_year': car_year, 'mileage': mileage,
                    'LTV': LTV, 'DTI': DTI, 'car_age': car_age, 'down_payment_pct': down_payment_pct,
                }
                result = engine.predict(features)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Вероятность дефолта (ML)", f"{result['probability']:.1%}")
                    risk_colors = {'low': '🟢 Низкий', 'medium': '🟡 Средний', 'high': '🔴 Высокий'}
                    st.metric("Класс риска", risk_colors[result['risk_class']])
                    
                    if result['risk_class'] == 'high':
                        st.error("⛔ Рекомендуется отказать или запросить доп. залог.")
                    elif result['risk_class'] == 'medium':
                        st.warning("⚠️ Кредит может быть одобрен с повышенной ставкой.")
                    else:
                        st.success("✅ Кредит может быть одобрен на стандартных условиях.")
                
                with c2:
                    st.subheader("🔍 Ключевые факторы риска")
                    if result['top_factors']:
                        factors_df = pd.DataFrame(result['top_factors'], columns=['Признак', 'Важность'])
                        factors_df['Признак'] = factors_df['Признак'].map({
                            'LTV': 'LTV', 'DTI': 'DTI', 'car_age': 'Возраст авто',
                            'credit_history_score': 'Кредитный рейтинг',
                            'down_payment_pct': 'Перв. взнос (%)', 'income': 'Доход'
                        })
                        st.bar_chart(factors_df.set_index('Признак'))

# ============================================================
# ВКЛАДКА 2: СРАВНЕНИЕ МОДЕЛЕЙ (ДАШБОРД)
# ============================================================
with tab2:
    st.header("📊 Дашборд сравнения моделей машинного обучения")
    st.markdown("*Сравнение 4 алгоритмов на несбалансированных данных с применением SMOTE*")
    
    try:
        comparator = load_comparator()
        X_test, y_test = load_test_data()
        
        df_comp = comparator.get_comparison_table()
        best_model = comparator.get_best_model_name()
        
        # 1. Таблица сравнения
        st.subheader("📋 Сводная таблица метрик")
        
        # Подсветка лучшей модели
        def highlight_best(s):
            is_best = s.name == df_comp[df_comp['PR-AUC'] == df_comp['PR-AUC'].max()].index[0]
            return ['background-color: #90EE90' if is_best else '' for _ in s]
        
        styled_df = df_comp.style.apply(highlight_best, axis=1).format({
            'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}',
            'ROC-AUC': '{:.3f}', 'PR-AUC': '{:.3f}', 'Business_Loss': '{:,.0f} ₽'
        })
        st.dataframe(styled_df, use_container_width=True)
        
        # 2. Визуальное сравнение метрик
        st.subheader("📈 Визуальное сравнение ключевых метрик")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Столбчатая диаграмма метрик
            metrics_to_plot = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
            fig_metrics = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for i, metric in enumerate(metrics_to_plot):
                fig_metrics.add_trace(go.Bar(
                    name=metric,
                    x=df_comp['model'],
                    y=df_comp[metric],
                    marker_color=colors[i % len(colors)]
                ))
            
            fig_metrics.update_layout(
                barmode='group',
                title="Сравнение метрик по моделям",
                yaxis_title="Значение метрики",
                xaxis_title="Модель",
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # Бизнес-убыток
            fig_loss = px.bar(
                df_comp, 
                x='model', 
                y='Business_Loss',
                title="Бизнес-убыток по моделям (меньше = лучше)",
                color='Business_Loss',
                color_continuous_scale='RdYlGn_r'
            )
            fig_loss.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # 3. ROC и PR кривые
        st.subheader("📉 ROC и Precision-Recall кривые")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC кривые
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                          line=dict(dash='dash', color='gray'), 
                                          name='Случайное угадывание'))
            
            for model_name in df_comp['model']:
                fpr, tpr, roc_auc = comparator.get_roc_curve_data(model_name, X_test, y_test)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{model_name} (AUC={roc_auc:.3f})',
                    line=dict(width=2)
                ))
            
            fig_roc.update_layout(
                title="ROC-кривые всех моделей",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # PR кривые (главное при дисбалансе!)
            fig_pr = go.Figure()
            baseline_pr = y_test.mean()
            fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline_pr, baseline_pr], 
                                         mode='lines', line=dict(dash='dash', color='gray'),
                                         name=f'Бейзлайн (доля дефолтов={baseline_pr:.3f})'))
            
            for model_name in df_comp['model']:
                precision, recall, ap = comparator.get_pr_curve_data(model_name, X_test, y_test)
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision, mode='lines',
                    name=f'{model_name} (AP={ap:.3f})',
                    line=dict(width=2)
                ))
            
            fig_pr.update_layout(
                title="Precision-Recall кривые (ключевая метрика при дисбалансе)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=450
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # 4. Детальный разбор выбранной модели
        st.subheader("🔍 Детальный разбор модели")
        
        selected_model = st.selectbox(
            "Выберите модель для анализа", 
            df_comp['model'].tolist(),
            index=df_comp['model'].tolist().index(best_model)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Метрики выбранной модели
            metrics = comparator.get_model_metrics(selected_model)
            st.markdown(f"### 📊 Метрики: **{selected_model}**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
            m2.metric("Recall", f"{metrics['Recall']:.3f}")
            m3.metric("F1", f"{metrics['F1']:.3f}")
            
            m4, m5, m6 = st.columns(3)
            m4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
            m5.metric("Precision", f"{metrics['Precision']:.3f}")
            m6.metric("Убыток", f"{metrics['Business_Loss']:,.0f} ₽")
        
        with col2:
            # Важность признаков
            st.markdown("### 🎯 Важность признаков")
            importance_df = comparator.get_feature_importance(selected_model)
            if importance_df is not None:
                top_n = importance_df.head(10)
                fig_imp = px.bar(
                    top_n, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title=f"Топ-10 важных признаков ({selected_model})",
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
        
        # 5. Выводы
        st.subheader("💡 Выводы дашборда")
        
        st.success(f"""
        **Лучшая модель:** `{best_model}`
        
        **Ключевые findings:**
        - **PR-AUC** (главная метрика при дисбалансе 6.8%): лучшая модель = **{df_comp['PR-AUC'].max():.3f}**
        - **Recall** (доля найденных дефолтов): лучшая модель = **{df_comp['Recall'].max():.3f}**
        - **Бизнес-убыток**: лучшая модель снижает убыток на **{(1 - df_comp['Business_Loss'].min() / df_comp['Business_Loss'].max()) * 100:.1f}%** по сравнению с худшей
        
        **Интерпретация:**
        - CatBoost/XGBoost с SMOTE показывают наилучшие результаты благодаря способности улавливать нелинейные зависимости
        - Logistic Regression — стабильный бенчмарк, но уступает ансамблям
        - Метрика **PR-AUC** корректнее ROC-AUC при сильном дисбалансе классов
        """)
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных для дашборда: {e}")
        st.info("💡 Убедитесь, что вы запустили `python -m src.train` и в папке models есть файлы .pkl и comparison.csv")

# ============================================================
# ВКЛАДКА 3: О СИСТЕМЕ
# ============================================================
with tab3:
    st.header("ℹ️ О системе")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Назначение")
        st.write("""
        Система предназначена для автоматизированной оценки вероятности дефолта 
        по автокредитам на основе характеристик заёмщика и предмета залога.
        """)
        
        st.subheader("🛠️ Технологический стек")
        st.code("""
        • Python 3.10
        • Scikit-learn, XGBoost, CatBoost
        • Imbalanced-learn (SMOTE)
        • Streamlit (веб-интерфейс)
        • Plotly (интерактивные графики)
        • Pandas, NumPy
        """)
    
    with col2:
        st.subheader("📊 Метрики качества")
        st.write("""
        - **PR-AUC** — главная метрика при дисбалансе классов
        - **Recall** — доля найденных дефолтов
        - **F1-score** — баланс Precision и Recall
        - **ROC-AUC** — общая способность модели ранжировать
        - **Business Loss** — финансовая оценка ошибок
        """)
        
        st.subheader("🏗️ Архитектура")
        st.write("""
        1. Уровень данных (Kaggle / синтетика)
        2. Уровень хранения (SQLite)
        3. Уровень обработки (StandardScaler + SMOTE)
        4. Уровень ML (4 модели)
        5. Уровень представления (Streamlit)
        """)

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "© 2026 | Курсовая работа: Сравнение производительности алгоритмов ML "
    "при работе с несбалансированными данными на примере прогнозирования дефолта автокредита"
    "</div>",
    unsafe_allow_html=True
)