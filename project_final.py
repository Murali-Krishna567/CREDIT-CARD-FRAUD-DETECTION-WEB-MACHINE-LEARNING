import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle

data = pd.read_csv('creditcard.csv')
#sns.jointplot(x='Time', y='Amount', data=data)
sns.relplot(x="Amount", y="Time", hue="Class", data=data)

#sns.relplot(x="Amount", y="Time", hue="Class", data=data)

class0 = data[data['Class'] == 0]
class1 = data[data['Class'] == 1]
temp = shuffle(class0)
d1 = temp.iloc[:2000, :]
frames = [d1, class1]
df_temp = pd.concat(frames)
#print(df_temp)

df_temp.info()

df = shuffle(df_temp)

df.to_csv('creditcardsampling.csv')

#sns.countplot('Class', data=df)

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X, Y = oversample.fit_resample(X, Y)
X = pd.DataFrame(X)
X.shape

Y = pd.DataFrame(Y)
Y.head()

names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
         'v17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

data = pd.concat([X, Y], axis=1)

d = data.values

data = pd.DataFrame(d, columns=names)

sns.countplot('Class', data=data)

data.describe()

data.info()

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.3, random_state=42)

"""# Feature Scaling"""

#cols = ['V22', 'V24', 'V25', 'V26', 'V27', 'V28']

scaler = StandardScaler()

frames = ['Time', 'Amount']

x = data[frames]

d_temp = data.drop(frames, axis=1)

temp_col = scaler.fit_transform(x)

scaled_col = pd.DataFrame(temp_col, columns=frames)

scaled_col.head()

d_scaled = pd.concat([scaled_col, d_temp], axis=1)

d_scaled.head()

y = data['Class']

d_scaled.head()

"""# Dimensionality Reduction"""

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
X_temp_reduced = pca.fit_transform(d_scaled)

names = ['Time', 'Amount', 'Transaction Method', 'Transaction Id', 'Location', 'Type of Card', 'Bank']

X_reduced = pd.DataFrame(data=X_temp_reduced, columns=names)
X_reduced.head()
print(X_reduced.head())

Y = d_scaled['Class']

new_data = pd.concat([X_reduced, Y], axis=1)
new_data.head()
new_data.shape

new_data.to_csv('finaldata.csv')

X_train, X_test, y_train, y_test = train_test_split(X_reduced, d_scaled['Class'], test_size=0.30, random_state=42)

X_train.shape, X_test.shape

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

import pickle
# Saving model to disk
pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
