# Lead Lifecycle Dashboard with Conversion Prediction and Priority Scoring
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# === Page Setup ===
st.set_page_config(page_title="Lead Lifecycle Dashboard", layout="wide")

# === Load and Prepare Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("lead_data.csv")
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Actual Date of Closure'] = pd.to_datetime(df['Actual Date of Closure'], errors='coerce')
    df['Lead Lifecycle (days)'] = (df['Actual Date of Closure'] - df['Created Date']).dt.days
    df['Lead Age (days)'] = (pd.Timestamp.today() - df['Created Date']).dt.days
    df['Activity'] = df['Activity'].fillna("Unknown")
    df['Converted'] = df['Actual Date of Closure'].notna().astype(int)
    return df

df = load_data()

# === Split Closed and Open Leads ===
closed = df.dropna(subset=['Created Date', 'Actual Date of Closure'])
closed = closed[(closed['Lead Lifecycle (days)'] >= 0) & (closed['Lead Lifecycle (days)'] <= 365)]
open_leads = df[df['Created Date'].notna() & df['Actual Date of Closure'].isna()]

# === Sidebar Filters ===
st.sidebar.header("Filters")
selected_status = st.sidebar.multiselect("Status", df['Status'].dropna().unique())
selected_product = st.sidebar.multiselect("Product", df['Product'].dropna().unique())
selected_type = st.sidebar.multiselect("Business Type", df['Business Type'].dropna().unique())
selected_activity = st.sidebar.multiselect("Activity", df['Activity'].dropna().unique())
lead_type = st.sidebar.selectbox("Lead Type", ["All", "Closed", "Open"])

# === Filter Application ===
def apply_filters(data):
    if selected_status:
        data = data[data['Status'].isin(selected_status)]
    if selected_product:
        data = data[data['Product'].isin(selected_product)]
    if selected_type:
        data = data[data['Business Type'].isin(selected_type)]
    if selected_activity:
        data = data[data['Activity'].isin(selected_activity)]
    return data

filtered_closed = apply_filters(closed)
filtered_open = apply_filters(open_leads)

if lead_type == "Closed":
    display_closed = filtered_closed
    display_open = pd.DataFrame()
elif lead_type == "Open":
    display_closed = pd.DataFrame()
    display_open = filtered_open
else:
    display_closed = filtered_closed
    display_open = filtered_open

# === Feature Engineering ===
def preprocess_features(df):
    df = df.copy()
    df['Created Month'] = df['Created Date'].dt.month
    df['Created Weekday'] = df['Created Date'].dt.weekday
    df['Is Weekend Created'] = df['Created Weekday'].isin([5, 6]).astype(int)
    return df

# === Train Regression Model ===
def train_regression(data):
    data = preprocess_features(data.dropna(subset=['Lead Lifecycle (days)']))
    features = ['Status', 'Product', 'Business Type', 'Activity', 'Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created']
    X = data[features]
    y = data['Lead Lifecycle (days)']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Status', 'Product', 'Business Type', 'Activity']),
        ('num', 'passthrough', ['Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created'])
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])

    model.fit(X, y)
    return model

# === Train Conversion Classifier ===
def train_classifier(data):
    data = preprocess_features(data)
    features = ['Status', 'Product', 'Business Type', 'Activity', 'Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created']
    X = data[features]
    y = data['Converted']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Status', 'Product', 'Business Type', 'Activity']),
        ('num', 'passthrough', ['Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created'])
    ])

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    clf.fit(X, y)
    return clf

reg_model = train_regression(closed)
clf_model = train_classifier(df)

# === Model Evaluation ===
def evaluate_regression(data, model):
    data = preprocess_features(data.dropna(subset=['Lead Lifecycle (days)']))
    features = ['Status', 'Product', 'Business Type', 'Activity', 'Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created']
    X = data[features]
    y = data['Lead Lifecycle (days)']
    preds = model.predict(X)
    return mean_absolute_error(y, preds), sqrt(mean_squared_error(y, preds)), r2_score(y, preds), y, preds

mae, rmse, r2, y_true, y_pred = evaluate_regression(closed, reg_model)

# === Predict Open Leads ===
def predict_open(data, reg_model, clf_model):
    data = preprocess_features(data)
    features = ['Status', 'Product', 'Business Type', 'Activity', 'Lead Age (days)', 'Created Month', 'Created Weekday', 'Is Weekend Created']
    reg_preds = reg_model.predict(data[features])
    clf_preds = clf_model.predict_proba(data[features])[:, 1]

    data['Predicted Lifecycle (days)'] = pd.Series(reg_preds).clip(lower=0)
    data['Predicted Closure Date'] = data['Created Date'] + pd.to_timedelta(data['Predicted Lifecycle (days)'].round(), unit='D')
    data['Conversion Probability'] = (clf_preds * 100).round(2)
    data['Lead Priority Score'] = (clf_preds * 100 + (365 - data['Lead Age (days)']).clip(lower=0) * 0.2).round(2)
    data['Conversion Category'] = pd.cut(data['Conversion Probability'],
        bins=[0, 40, 70, 100],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True)
    return data

predicted_open = predict_open(open_leads, reg_model, clf_model)

# === Dashboard Layout ===
st.title("📊 Lead Lifecycle Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Leads (Closed)", len(display_closed))
col2.metric("Avg Lifecycle (Days)", round(display_closed['Lead Lifecycle (days)'].mean(), 2) if not display_closed.empty else "N/A")
col3.metric("Open Leads", len(display_open))

st.markdown("---")
st.markdown("### 🧠 Model Performance")
st.markdown(f"- **MAE:** {mae:.2f} days  |  **RMSE:** {rmse:.2f}  |  **R² Score:** {r2:.2f}")
st.write("Conversion Rate:", df['Converted'].mean())

# === Closed Leads Visualizations ===
if not display_closed.empty:
    st.subheader("📈 Lifecycle Distribution (Closed)")
    fig1, ax1 = plt.subplots(figsize=(5, 2.5))
    sns.histplot(display_closed['Lead Lifecycle (days)'], bins=30, kde=True, ax=ax1, color='skyblue')
    st.pyplot(fig1)

    for label in ['Status', 'Product', 'Business Type', 'Activity']:
        st.subheader(f"📊 Avg Lifecycle by {label}")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        avg = display_closed.groupby(label)['Lead Lifecycle (days)'].mean().sort_values()
        sns.barplot(x=avg.index, y=avg.values, ax=ax, palette='Blues_d')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    st.subheader("🔍 Actual vs Predicted")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.scatter(y_true, y_pred, alpha=0.6)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)

# === Open Leads Table ===
if not display_open.empty:
    filtered_predicted_open = predicted_open[predicted_open['Form Id'].isin(display_open['Form Id'])]

    st.subheader("🤖 Open Leads Predictions")
    st.dataframe(filtered_predicted_open[[
        'Form Id', 'Created Date', 'Status', 'Product', 'Business Type', 'Activity',
        'Lead Age (days)', 'Predicted Lifecycle (days)', 'Predicted Closure Date',
        'Conversion Probability', 'Conversion Category', 'Lead Priority Score']])

    st.subheader("📅 Predicted Closures Calendar")
    cal_df = filtered_predicted_open.dropna(subset=['Predicted Closure Date'])
    cal_df['Week'] = cal_df['Predicted Closure Date'].dt.to_period('W').dt.start_time
    forecast = cal_df.groupby('Week').size().reset_index(name='Predicted Closures')
    fig3, ax3 = plt.subplots(figsize=(6, 2.5))
    sns.barplot(data=forecast, x='Week', y='Predicted Closures', color='orange', ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)

    st.subheader("📈 Conversion Probability and Priority Score Trends")
    sorted_df = filtered_predicted_open.sort_values("Created Date")

    fig4, (ax4, ax5) = plt.subplots(nrows=2, ncols=1, figsize=(7, 4), sharex=True)
    sns.lineplot(data=sorted_df, x="Created Date", y="Conversion Probability", ax=ax4, color="green")
    ax4.set_title("Conversion Probability over Time")

    sns.lineplot(data=sorted_df, x="Created Date", y="Lead Priority Score", ax=ax5, color="purple")
    ax5.set_title("Lead Priority Score over Time")

    for ax in (ax4, ax5):
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("📊 Conversion Category Distribution")
    fig5, ax6 = plt.subplots(figsize=(4, 2.5))
    category_counts = filtered_predicted_open['Conversion Category'].value_counts().sort_index()
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='Set2', ax=ax6)
    ax6.set_ylabel("Number of Leads")
    st.pyplot(fig5)
    # === Activity-Based Insights ===
    # === Activity-Based Time Insights ===
    st.markdown("## 🔍 Activity Time-Based Insights")

    # Ensure datetime columns are parsed
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Demo date'] = pd.to_datetime(df['Demo date'], errors='coerce')
    df['Quote Sent Date'] = pd.to_datetime(df['Quote Sent Date'], errors='coerce')

    # 1. Activity Timeline
    st.subheader("1. Activity Timeline (Count by Date)")
    activity_timeline = df.groupby(['Date', 'Activity']).size().unstack(fill_value=0)
    fig_at1, ax_at1 = plt.subplots(figsize=(8, 3))
    activity_timeline.rolling(7).mean().plot(ax=ax_at1)  # 7-day rolling average for smoother trend
    ax_at1.set_title("Activity Trends Over Time")
    ax_at1.set_xlabel("Date")
    ax_at1.set_ylabel("Count")
    ax_at1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    st.pyplot(fig_at1)

    # 2. Conversion-Oriented Activity (Demo & Quote Dates)
    st.subheader("2. Demos and Quotes Sent Over Time")
    demo_ts = df['Demo date'].dropna().dt.date.value_counts().sort_index()
    quote_ts = df['Quote Sent Date'].dropna().dt.date.value_counts().sort_index()

    fig_at2, ax_at2 = plt.subplots(figsize=(8, 3))
    pd.Series(demo_ts).rolling(7).mean().plot(ax=ax_at2, label='Demos', marker='o')
    pd.Series(quote_ts).rolling(7).mean().plot(ax=ax_at2, label='Quotes Sent', marker='x')
    ax_at2.set_title("Demo & Quote Trends Over Time")
    ax_at2.set_xlabel("Date")
    ax_at2.set_ylabel("Count (7-Day MA)")
    ax_at2.legend()
    st.pyplot(fig_at2)



    # 3. Top Activities Timeline
    st.subheader("3. Top Activities Timeline (Top 5 Only)")
    top_activities = df['Activity'].value_counts().head(5).index
    top_df = df[df['Activity'].isin(top_activities)].copy()
    top_df = top_df.groupby(['Date', 'Activity']).size().unstack(fill_value=0)

    fig_at3, ax_at3 = plt.subplots(figsize=(8, 3))
    top_df.rolling(7).mean().plot(ax=ax_at3)
    ax_at3.set_title("Top 5 Activities Over Time")
    ax_at3.set_xlabel("Date")
    ax_at3.set_ylabel("Count (7-Day MA)")
    ax_at3.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    st.pyplot(fig_at3)

    # 4. Next Actions Over Time
    st.subheader("4. Top Next Actions Over Time")
    df['Date for Next action'] = pd.to_datetime(df['Date'], errors='coerce')
    top_next_actions = df['Next Action'].value_counts().head(5).index
    next_action_df = df[df['Next Action'].isin(top_next_actions)].copy()
    next_action_ts = next_action_df.groupby(['Date for Next action', 'Next Action']).size().unstack(fill_value=0)

    fig_at4, ax_at4 = plt.subplots(figsize=(8, 3))
    next_action_ts.rolling(7).mean().plot(ax=ax_at4)
    ax_at4.set_title("Top 5 Next Actions Over Time")
    ax_at4.set_xlabel("Date")
    ax_at4.set_ylabel("Count (7-Day MA)")
    ax_at4.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    st.pyplot(fig_at4)





    



