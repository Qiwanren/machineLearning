from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_moons
X,y = make_moons(noise=.2)
print(X)
print('--------------------------------------------------')
print(y)

mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.show()