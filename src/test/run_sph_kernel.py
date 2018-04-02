from src.sph.kernel import *

A2D = np.asarray([[1, 3, 5],
                [2, 4, 6]])
print("2D poly6:\n", W_poly6_2D(A2D,5))

A3D = np.asarray([[1, 3, 5, 7 ],
                [2, 4, 6, 8],
                [1, 1, 1, 1,]])

print("3D poly6:\n",W_poly6_3D(A3D,6))
print("3D poly6 gradient:\n",nablaW_poly6_3D(A3D,6))
print("3D poly6 Laplacian:\n",nabla2W_poly6_3D(A3D,6))

h=10
import matplotlib.pyplot as plt
Xnorms = np.mgrid[-h:h:0.1]
X = np.c_[Xnorms.T, np.zeros(Xnorms.T.shape)]
W = W_poly6_2D(X.T,h)


print("2d sum of W coefficients: ", np.sum(W))

plt.plot (X,  W)
plt.show()
