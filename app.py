import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Get the data
housing_data = pd.read_csv('usa_housing.csv')

# Check the head
housing_data.head()


X = housing_data.drop(['Price', 'Address'], axis=1)
Y = housing_data['Price']

# Split the data into a training set and a testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


# Create and Train the Model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)


# Predict it boiii
predictions = linear_model.predict(X_test)

# Check the results
plt.scatter(Y_test, predictions)
plt.show()
