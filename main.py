import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('Admission_Predict.csv')

df = data.copy()
scale = StandardScaler()
cols = df.columns[1:7]
df[cols] = scale.fit_transform(df[cols])

temp_list = np.array(df)
x = range(1,6)
temp = []
for i in x:
    model = KMeans(n_clusters=i)
    model.fit(temp_list)
    temp.append(model.inertia_)
plt.plot(x, temp, '-o')
plt.show()

pca = PCA(n_components=2)
pc1 = pca.fit_transform(temp_list[:,1:])
x_data = pc1[:,0]
y_data = pc1[:,1]
model=KMeans(n_clusters=3)
model.fit(pc1)
sns.scatterplot(x_data,y_data,hue=model.labels_)
plt.show()
svc = SVC(kernel='linear')

svc.fit(temp_list, temp_list[cols])

new_df = df['Chance of Admit '].copy()
new_df[new_df>=0.85] = 1
new_df[new_df<0.85] = 0
df['Chance of Admit '] = new_df
data_y = df['Chance of Admit ']
data_x = df
data_x = data_x.drop(columns=['Chance of Admit '])
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.30,random_state=109) 

svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
