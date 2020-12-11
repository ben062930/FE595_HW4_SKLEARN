from sklearn.datasets import load_boston, load_iris, load_wine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


boston_dataset = load_boston()
#print(boston_dataset.keys())
#boston_dataset.DESCR

# transfer array to data frame
boston_feature = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
#boston_feature.head()

boston_target = pd.DataFrame(boston_dataset.target)
#boston_target.head()

#check null values
#boston_feature.isnull().sum()
#boston_target.isnull().sum()
#no null value, pass

line_model = LinearRegression()
line_model.fit(boston_feature,boston_target)

#basic statistics
r_square = line_model.score(boston_feature, boston_target)
print('coefficient of determination:', r_square)
print('intercept:', line_model.intercept_)
print('slope:', line_model.coef_)

coef =  pd.DataFrame(line_model.coef_, columns = boston_dataset.feature_names)
#coef._stat_axis.values.tolist()
coef.index = ['coef']

#from high to low ranking
rank = coef.abs().sort_values(axis=1, by=['coef'], ascending=False)
variable_list = rank.columns.values.tolist()

#from greatest impact ot lowest impact
print(coef[variable_list])


###since no evaluation is required, I don't divide the original data into test and train subgrop
#%%


iris = load_iris()
#print(iris.keys())
iris_data = pd.DataFrame(iris['data'])
#iris_data.head()

distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(iris_data)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Heuristic - IRIS')
plt.savefig('IRIS.png')
plt.show()

### we can easily find that K = 3 is the elbow
### When K comes to 3, the Distortions becomes flat
### K=4 may be good, too but K=3 is more optimal

wine = load_wine()
#print(wine.keys())
wine_data = pd.DataFrame(wine['data'])
#wine.head()

distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(wine_data)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Heuristic - Wine')
plt.savefig('Wine.png')
plt.show()

### we can easily find that K = 3 is the elbow
### When K comes to 3, the Distortions becomes flat
### K=4 may be good, too but K=3 is more optimal
