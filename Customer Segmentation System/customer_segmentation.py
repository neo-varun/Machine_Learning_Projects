import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("Customer Segmentation System/customers.csv")

df = df.dropna()
df = df.drop("CustomerID", axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

plt.scatter(df["Income"], df["SpendingScore"], c=df["Cluster"])
plt.xlabel("Income")
plt.ylabel("SpendingScore")
plt.show()

print(df.groupby("Cluster").mean())
