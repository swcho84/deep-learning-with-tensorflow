import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

'''
weight initialization
'''
def weight_variable(shape, name=None):
    return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)

np.random.seed(0)

'''
데이터를 생성한다
toy problem 들은 weight initialization 과 batch normalization 으로 성능향상 가능
'''
N = 300
X, y = datasets.make_moons(N, noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

'''
모델을 생성한다
'''
model = Sequential()
model.add(Dense(5, input_dim=2, kernel_initializer=weight_variable))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dense(1, kernel_initializer=weight_variable))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

'''
모델을 학습시킨다
'''
model.fit(X_train, y_train, epochs=500, batch_size=20)

'''
예측 정확도를 평가한다
'''
loss_and_metrics = model.evaluate(X_test, y_test)
print(loss_and_metrics)

'''
plotting the decision boundary
'''
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.1
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.get_cmap('Spectral'))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap('Spectral'))

# Predict and plot
plot_decision_boundary(lambda x: model.predict_classes(x, batch_size=20))
plt.title("Decision Boundary")
plt.show()
