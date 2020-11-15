import pandas as pd

AnimalClass = pd.read_csv('animal_classes.csv')
AnimalTest = pd.read_csv('animals_test.csv')
AnimalTrain = pd.read_csv('animals_train.csv')

print(AnimalClass.head())
print()
print(AnimalTrain.head())
print()
print(AnimalTest.head())


# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(
#     AnimalTrain.class_number.values.reshape(-1, 1),  AnimalTest.legs.values
# )

x_train = AnimalTrain

# # print(x_train.shape)
# # print(x_test.shape)
# # print(y_train.shape)
# # print(y_test.shape)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X = x_train, y = y_train)

# for i, name in enumerate(cali.feature_names):
#     print(f"{name:>10}: {linear_regression.coef_[i]}")

# predicted = linear_regression.predict(x_test)
# print(predicted[:5])    # View the first 5 predictions

# expected = y_test 
# print(expected[:5])     # view the first 5 expected target values

# #Create a DataFrame containing columns for the expected and predicted values:

# df = pd.DataFrame()

# df["Expected"] = pd.Series(expected)
# df["Predicted"] = pd.Series(predicted)
