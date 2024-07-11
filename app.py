import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Define numeric features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Churn Prediction')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
def user_input_features():
    tenure = st.sidebar.number_input('Tenure', min_value=0, step=1)
    monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=0.0, step=0.1)
    total_charges = st.sidebar.number_input('Total Charges', min_value=0.0, step=0.1)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': 0 if gender == 'Male' else 1,
        'Partner': 0 if partner == 'No' else 1,
        'Dependents': 0 if dependents == 'No' else 1,
        'PhoneService': 0 if phone_service == 'No' else 1,
        'MultipleLines': 0 if multiple_lines == 'No' else 1,
        'InternetService': 0 if internet_service == 'DSL' else (1 if internet_service == 'Fiber optic' else 2),
        'OnlineSecurity': 0 if online_security == 'No' else 1,
        'OnlineBackup': 0 if online_backup == 'No' else 1,
        'DeviceProtection': 0 if device_protection == 'No' else 1,
        'TechSupport': 0 if tech_support == 'No' else 1,
        'StreamingTV': 0 if streaming_tv == 'No' else 1,
        'StreamingMovies': 0 if streaming_movies == 'No' else 1,
        'Contract': 0 if contract == 'Month-to-month' else (1 if contract == 'One year' else 2),
        'PaperlessBilling': 0 if paperless_billing == 'No' else 1,
        'PaymentMethod': 0 if payment_method == 'Electronic check' else (1 if payment_method == 'Mailed check' else (2 if payment_method == 'Bank transfer (automatic)' else 3))
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Scale numeric features
input_data[numeric_features] = scaler.transform(input_data[numeric_features])

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]

st.subheader('Prediction')
st.write('Churn Prediction:', 'Yes' if prediction == 1 else 'No')
st.write('Prediction Probability:', prediction_proba)

# Visualization
st.subheader('Visualizations')

# Feature importances
feature_importances = pd.DataFrame({
    'Feature': input_data.columns,
    'Importance': model.coef_[0]
})

fig, ax = plt.subplots()
sns.barplot(data=feature_importances.sort_values(by='Importance', ascending=False), x='Importance', y='Feature', ax=ax)
plt.title('Feature Importances')
st.pyplot(fig)

# Confusion matrix
st.subheader('Confusion Matrix')
y_test = [0, 1]  # replace with actual y_test data
y_pred = [0, 1]  # replace with actual y_pred data
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Confusion Matrix')
st.pyplot(fig)

# ROC Curve
st.subheader('ROC Curve')
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
st.pyplot(fig)
