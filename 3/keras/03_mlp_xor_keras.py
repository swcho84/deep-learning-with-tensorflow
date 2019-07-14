import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

np.random.seed(123)

'''
데이터를 생성한다
'''
# XORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
모델을 설정한다
'''
model = Sequential()

# 입력층-은닉층
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 은닉층-출력층
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
모델을 학습시킨다
'''
model.fit(X, Y, epochs=4000, batch_size=4)

'''
학습 결과를 확인한다
'''
classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
print('W:', model.get_weights()[0])
print('b:', model.get_weights()[1])

Weight = model.get_weights()[0] # Optimized Weight 
Bias = model.get_weights()[1]   # Optimized Bias

'''
data visualization
'''
plot_x = np.array([X[:, 0].min(), X[:, 0].max()]).reshape(2, 1)
Bias = Bias.reshape(2, 1)
temp1 = ((Weight[1].reshape(2, 1) * plot_x[0]) + (Bias))
temp2 = ((Weight[1].reshape(2, 1) * plot_x[1]) + (Bias))
plot_y1 = np.zeros(2).reshape(2, 1)
plot_y2 = np.zeros(2).reshape(2, 1)
for i in range(2):
    plot_y1[i] = (-1/(Weight[0][i])) * temp1[i][0]
    plot_y2[i] = (-1/(Weight[0][i])) * temp2[i][0]
    
data = np.hstack([X, Y])
colors = {0:'red', 1:'blue', 2:'green'}
plt.figure()
rows_data, cols_data = data.shape
for i in range(rows_data):
    if (data[i][2] == 1):
        plt.scatter(data[i][0], data[i][1], c=colors[0])
    else:
        plt.scatter(data[i][0], data[i][1], c=colors[1])
plt.plot(plot_x, [plot_y1[0][0], plot_y2[0][0]])
plt.plot(plot_x, [plot_y1[1][0], plot_y2[1][0]])
plt.show()
