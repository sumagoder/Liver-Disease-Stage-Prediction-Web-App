import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Liver Disease Classifier App", layout="wide")
st.title("🩺 Liver Disease Prediction - Interactive ML App")

uploaded_file = st.file_uploader("📂 Upload your liver dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtypes != 'object':
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df.replace({
        'Sex': {'M': 1, 'F': 0},
        'Edema': {'N': 0, 'S': -1, 'Y': 1},
        'Ascites': {'Y': 1, 'N': 0},
        'Hepatomegaly': {'Y': 1, 'N': 0},
        'Spiders': {'Y': 1, 'N': 0},
        'Drug': {'D-penicillamine': 0, 'Placebo': 1}
    }, inplace=True)

    df.drop(['N_Days', 'Status'], axis=1, inplace=True, errors='ignore')

    # EDA section
    st.subheader("📊 Exploratory Data Analysis")
    with st.expander("Show Distribution Plots"):
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(df[col], kde=True, ax=ax[0])
            ax[0].set_title(f"Distribution of {col}")
            sns.boxplot(x=df[col], ax=ax[1])
            ax[1].set_title(f"Boxplot of {col}")
            st.pyplot(fig)

    with st.expander("Show Pie Charts for Categorical Features"):
        cat_cols = ['Stage', 'Sex', 'Edema', 'Ascites', 'Hepatomegaly', 'Spiders', 'Drug']
        for col in cat_cols:
            if col in df.columns:
                labels = df[col].value_counts().index
                sizes = df[col].value_counts().values
                explode = [0.1] + [0] * (len(labels)-1)
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)
                ax.set_title(f"{col} Distribution")
                st.pyplot(fig)

    # Feature Engineering
    X = df.drop('Stage', axis=1)
    y = df['Stage']

    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'KNN (k=17)': KNeighborsClassifier(n_neighbors=17),
        'SGD': SGDClassifier(),
        'Passive Aggressive': PassiveAggressiveClassifier(),
        'Ridge': RidgeClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=1000),
        'Extra Trees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'HistGradientBoosting': HistGradientBoostingClassifier(),
        'Gaussian NB': GaussianNB(),
        'Bernoulli NB': BernoulliNB(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'LightGBM': LGBMClassifier(),
        'XGBoost': XGBClassifier(eval_metric='mlogloss')
    }

    selected_model = st.selectbox("🤖 Choose a model to train", list(models.keys()))
    run = st.button("Train and Evaluate")

    if run:
        model = models[selected_model]
        model.fit(X_train, y_train)
        st.session_state['trained_model']=model
        y_pred = model.predict(X_test)

        st.subheader("📈 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔍 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.success(f"{selected_model} trained and evaluated successfully.")

    # Optional: Feature importance
    if selected_model in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'CatBoost', 'LightGBM', 'XGBoost'] and run:
        st.subheader("🧠 Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
        st.pyplot(fig)

    # Manual prediction input form
    with st.expander("📥 Try Manual Prediction"):
        manual_input = {}
        for col in X.columns:
            val = st.number_input(f"Enter value for {col}", value=float(df[col].median()))
            manual_input[col] = val
        if st.button("Predict on Manual Input"):
            if 'trained_model' not in st.session_state:
                st.warning("Please train a model first using the 'Train and Evaluate button.")
            else:
                user_df = pd.DataFrame([manual_input])
                user_df = pd.DataFrame(scaler.transform(user_df), columns=X.columns)
                prediction = st.session_state.trained_model.predict(user_df)[0]
                st.success(f"Predicted Stage: {prediction}")

else:
    st.warning("Upload a dataset to get started.")


