import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/DEON/PycharmProjects/Iris_Classification/data/iris.csv")
# Basic info
print(df.head())
print(df.info())
print(df.describe())
print(df['species'].value_counts())

# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()