from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs

X,y = make_blobs(random_state=42)
print(X)
print('--------------------------------------------------')
print(y)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.show()