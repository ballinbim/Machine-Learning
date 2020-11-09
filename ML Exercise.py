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

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.1)
sns.set_style("whitegrid")
sns.pairplot(data = cali_df, vars = cali_df.columns[0:4])

# for feature in cali.feature_names:
#     plt.figure(figsize = (8,4.5)) # 8"-by-4.5" figure
#     sns.scatterplot(
#         data = sample_df,
#         x = feature,
#         y = "MedHouseValue",
#         hue = "MedHouseValue",
#         palette="cool",
#         legend = False,
#     )
plt.show()
