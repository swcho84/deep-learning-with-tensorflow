import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import  sklearn
from    sklearn import datasets, linear_model

np.random.seed(0)  # 난수 시드

M = 2      # 입력 데이터의 차원
K = 3      # 클래스 수
n = 100    # 각 클래스에 있는 데이터 수
N = n * K  # 모든 데이터 수

'''
데이터 생성
'''
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

'''
모델을 설정한다
'''
model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

'''
모델을 학습시킨다
'''
minibatch_size = 50
model.fit(X, Y, epochs=20, batch_size=minibatch_size)

'''
학습 결과를 확인한다
'''
X_, Y_ = shuffle(X, Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=minibatch_size)
print('classified:')
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print('output probability:')
print(prob)
print('W:', model.get_weights()[0])
print('b:', model.get_weights()[1])

Weight = model.get_weights()[0] # Optimized Weight 
Bias = model.get_weights()[1]   # Optimized Bias

'''
data visualization using grid
'''
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
# def plot_decision_boundary(pred_func):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 0.1
    
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
#     # Predict the function value for the whole gid
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.get_cmap('Spectral'))
#     plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.get_cmap('Spectral'))

# # Predict and plot
# plot_decision_boundary(lambda x: model.predict_classes(x, batch_size=300))
# plt.title("Decision Boundary")
# plt.show()

'''
data visualization using lines
'''
plot_x = np.array([X[:, 0].min(), X[:, 0].max()]).reshape(2, 1)
Bias = Bias.reshape(3, 1)
temp1 = ((Weight[1].reshape(3, 1) * plot_x[0]) + (Bias))
temp2 = ((Weight[1].reshape(3, 1) * plot_x[1]) + (Bias))
plot_y1 = np.zeros(3).reshape(3, 1)
plot_y2 = np.zeros(3).reshape(3, 1)
for i in range(K):
    plot_y1[i] = (-1/(Weight[0][i])) * temp1[i][0]
    plot_y2[i] = (-1/(Weight[0][i])) * temp2[i][0]

data = np.hstack([X, Y])
colors = {0:'red', 1:'blue', 2:'green'}
plt.figure()
rows_data, cols_data = data.shape
for i in range(rows_data):
    if (data[i][2] == 1):
        plt.scatter(data[i][0], data[i][1], c=colors[0])
    elif (data[i][3] == 1):
        plt.scatter(data[i][0], data[i][1], c=colors[1])
    elif (data[i][4] == 1):
        plt.scatter(data[i][0], data[i][1], c=colors[2])
plt.plot(plot_x, [plot_y1[0][0], plot_y2[0][0]])
plt.plot(plot_x, [plot_y1[1][0], plot_y2[1][0]])
plt.plot(plot_x, [plot_y1[2][0], plot_y2[2][0]])
plt.show()