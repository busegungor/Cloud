import pandas as pd
import numpy as np
import pickle


df = pd.read_csv('fish.csv')

X = np.array(df.iloc[:, 0:3])
y = np.array(df.iloc[:, 3:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.ravel())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.svm import SVC
sv = SVC(kernel = 'linear').fit(X_train, y_train.ravel())

pickle.dump(sv, open('fish.pkl', 'wb'))
