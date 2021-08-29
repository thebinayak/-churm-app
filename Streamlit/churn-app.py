# Source : https://builtin.com/machine-learning/streamlit-tutorial

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64                   # conda install -c conda-forge pybase64
import seaborn as sns           #conda install seaborn
import matplotlib.pyplot as plt # conda install -c conda-forge matplotlib

st.write('''
 # Churn Prediction App
 
 Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with
the company. 

This app predicts the probability of a customer churning using Telco Customer data. Here
customer churn means the customer does not make another purchase after a period of time.
''')
# streamlit run churn-app.py


 ## SECOND : Modify app so that users can download the data the trained their model

df_selected = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df_selected.head())

df_selected_all = df_selected[['gender', 'Partner', 'Dependents', 'PhoneService','tenure', 'MonthlyCharges']].copy() #, 'target'


# Define a function that allows us to download the read-in data

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytesconversions
    href = f'<a href="data:file/csv;base64,{b64}" download="WA_Fn-UseC_-Telco-Customer-Churn.csv">Download CSV File</a>'
    #<a> and </a> is an anchor element used to create hyperlink.
    return href


# Specify the showPyplotGlobalUse deprecation warning as False
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)

# Categorical Input Select Box and Numerical Input Slider
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('gender', ('Male', 'Female'))
        PaymentMethod = st.sidebar.selectbox('PaymenrMethod', ('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 18.0)
        tenure = st.sidebar.slider('tenure', 0.0, 72.0, 0.0)

        data = {
            'gender'        : [gender],
            'PaymentMethod' : [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'tenure'        : [tenure],
            }
        features = pd.DataFrame(data)
        return features

    input_df = user_input_features()

# Display the output of the model (use default input and output incase the user does not select any)
churn_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Churn'])
df = pd.concat([input_df, churn], axis=0)

# Encode our features:
encode = ['gender', 'PaymentMethod']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) #Convert categorical variable into dummy/indicator variables.
    df = pd.concat([df, dummy], axis=1) # Add onehotencoded (from pd.get_dummies) variable to the df
    del df[col] # Delete original variable

df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

# Select the features we want to display:

features = ['MonthlyCharges', 'tenure', 'gender_Female', 'gender_Male',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
df = df[features]

# Finally, we display the default input using the write method:
# Displays the user input features

st.subheader('User Input features')
print(df.columns)
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Make predictions and display them either using the default input or the user input.
# First, we need to read in our saved model, which is in a Pickle file:
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# Generate binary scores and prediction probabilities:
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# And write the output:
st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])

st.write(churn_labels[prediction])
st.subheader('Prediction Probability')
st.write(prediction_proba)