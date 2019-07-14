import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

np.random.seed(0)  # 난수 시드

'''
모델 설정
'''
model = Sequential([
    Dense(input_dim=2, units=1),         # Keras 2
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

'''
모델 학습
'''
# OR 게이트
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

model.fit(X, Y, epochs=200, batch_size=1)      # Keras 2

'''
학습 결과를 확인한다
'''
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
print('W:', model.get_weights()[0])
print('b:', model.get_weights()[1])

Weight = model.get_weights()[0] # Optimized Weight 
Bias = model.get_weights()[1]   # Optimized Bias

plot_x = [X[:, 0].min() - .5, X[:, 0].max() + .5]
plot_y = (-1/(Weight[0])) * ((Weight[1]) * (plot_x) + (Bias))
print(plot_x)
print(plot_y)

'''
data visualization
'''
data = np.hstack([X, Y])
colors = {0:'red', 1:'blue', 2:'green'}
plt.figure()
rows_data, cols_data = data.shape
for i in range(rows_data):
    if (data[i][2] == 1):
        plt.scatter(data[i][0], data[i][1], c=colors[0])
    else:
        plt.scatter(data[i][0], data[i][1], c=colors[1])
plt.plot(plot_x, plot_y)
plt.show()
