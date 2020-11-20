import pandas as pd

AnimalClass = pd.read_csv('animal_classes.csv')
AnimalTest = pd.read_csv('animals_test.csv')
AnimalTrain = pd.read_csv('animals_train.csv')

train_df = AnimalTrain.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
target = AnimalTrain.iloc[:, [16]]

# print(AnimalClass.head())
# print(AnimalTrain)
# # print(AnimalTest.head())
# print(train_df)
# print(target)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_df,  target)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

test_df = AnimalTest.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
print(test_df)

# load the training data into the model using the fit method

knn.fit(X = x_train, y = y_train)

predicted = knn.predict(X = x_test)
expected = y_test

# print(predicted[:20])
# print(expected[:20])

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_true= expected, y_pred= predicted)
# print(cf)



