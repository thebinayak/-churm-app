import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(df_churn)

#pd.set_option('display.max_columns', None)
#pd.set_option('display_rows', None)
# df_churn = pd.read_csv()
df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn']].copy()
print(df_churn.head())

''' 
Let's store a copy of our data frame in a new variable called df and replace missing values with zero:
'''
df = df_churn.copy()
df.fillna(0, inplace=True)

'''
Let's create machine-readable dummy variable for our categorical columns Gender and PaymentMethod
'''
encode = ['gender', 'PaymentMethod']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) # Make dummy column
    df = pd.concat([df, dummy], axis=1) # Concatinate the dummy column in original dataframe
    del df[col] # delete original gender and PaymentMethod columns

print(df.head())
'''
Map the churm column values to binary values (Yes to 1 and No to 0)
'''
import numpy as np
df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)

# Define the input and output
X = df.drop('Churn', axis=1) # axis=1 is column
y = df['Churn']

'''
Define an instance of RandForestClassifier and fit our model to the data
'''
clf = RandomForestClassifier()
clf.fit(X,y)

# Save the model as Pickle file
pickle.dump(clf,open('churn_clf.pkl', 'wb'))
