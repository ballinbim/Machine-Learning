from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing() #bunch object
# print(cali.DESCR)

print(cali.data.shape)
print(cali.target.shape)
print(cali.feature_names)

import pandas as pd
pd.set_option("precision", 4)
pd.set_option("max_columns", 9) #display up to 9 columns in DataFrame 
pd.set_option("display.width", None) # auto-detect the display width 

cali_df = pd.DataFrame(cali.data, columns = cali.feature_names)
cali_df["MedHouseValue"] = pd.Series(cali.target)
print(cali_df.head()) 

sample_df = cali_df.sample(frac = 0.1, random_state = 17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 2)
sns.set_style("whitegrid")

for feature in cali.feature_names:
    plt.figure(figsize = (8,4.5)) # 8"-by-4.5" figure
    sns.scatterplot(
        data = sample_df,
        x = feature,
        y = "MedHouseValue",
        hue = "MedHouseValue",
        palette="cool",
        legend = False,
    )
# plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    cali.data, cali.target, random_state = 11
)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X = x_train, y = y_train)

for i, name in enumerate(cali.feature_names):
    print(f"{name:>10}: {linear_regression.coef_[i]}")

predicted = linear_regression.predict(x_test)
print(predicted[:5])    # View the first 5 predictions

expected = y_test 
print(expected[:5])     # view the first 5 expected target values

#Create a DataFrame containing columns for the expected and predicted values:

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

# plot the data as a scatter plot with expected (target)
# prices along the x-axis and the predicted prices along the y-axis:

import matplotlib.pyplot as plt2
figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(data = df, x = "Expected", y = "Predicted",
hue = "Predicted", palette= "cool", legend= False)

# Set the x- and y- axes' limits to use the same scale along both axes:
start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)

# The following snippet displays a line between the points represent
# the lower-left corner of the graph (start, start) and the upper-right
# corner of the graph (end, end). The third argument ('k--') indicates
# the line's style. The letter k represents the color black, and
# the -- indicates that plot should draw a dashed line:

line = plt2.plot([start, end], [start, end], "k--")
plt2.show()