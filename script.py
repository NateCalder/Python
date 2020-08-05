import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':1, 'male':0})

# Fill the nan values in the age column
passengers = passengers.fillna(value={'Age': passengers.Age.mean()})

# Create a first class column
passengers['FirstClass'] = passengers.Pclass.apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers.Pclass.apply(lambda x: 1 if x == 2 else 0)
print(passengers.head(20))


# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, train_size=0.8, test_size=0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
tfeatures = scaler.fit_transform(x_train)
stfeatures = scaler.transform(x_test)


# Create and train the model
model=LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the train data
sc = model.score(x_train, y_train)
sctest = model.score(x_test, y_test)
print(sc)

# Score the model on the test data
sctest = model.score(x_test, y_test)
print(sctest)

# Analyze the coefficients
print(model.coef_)
#Age appears to be the most important coefficient to the model at -0.034.

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Me = np.array([0.0,24.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Me])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
#print(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
