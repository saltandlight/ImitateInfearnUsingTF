# Lesson8 연습 1: XOR 게이트에 대한 신경망

## Logistic Regression 모델

```python
import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y_data = np.array([[0],
                   [1],
                   [1], [0]], dtype=np.float32)

# X, Y는 추후 선언

W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

@tf.function
def hypo(X):
    return tf.sigmoid(tf.matmul(X, W) + b)

@tf.function
def cost(X, Y):
    hyp = hypo(X)
    return -tf.reduce_mean(Y * tf.math.log(hyp) + (1-Y) * tf.math.log(1-hyp))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 모델의 정확도 계산
@tf.function
def predict(X):
    hyp = hypo(X)
    return tf.cast(hyp > 0.5, dtype=tf.float32)

@tf.function
def accuracy(X, Y):
    pred = predict(X)
    return tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

X = tf.Variable(x_data, dtype=tf.float32)
Y = tf.Variable(y_data, dtype=tf.float32)

for step in range(10001):
    with tf.GradientTape() as tp:
        cost_val = cost(X, Y)

    train = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    if step % 1000 == 0:
        print(f'step = {step:06}, cost: {cost_val}, weight: {tf.squeeze(W.numpy())}')

h = hypo(X)
p = predict(X)
a = accuracy(X, Y)
print(f'Hypothesis : {tf.squeeze(h).numpy()}')
print(f'Correct : {tf.squeeze(p).numpy()}')
print(f'Accuracy: {a*100:.6}%')
```

## Neural Network 모델

- shape 모양 변경 -> 계산값 더 명확해짐

```python
import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y_data = np.array([[0],
                   [1],
                   [1], [0]], dtype=np.float32)

# Neural Network 적용
W1 = tf.Variable(tf.random.normal([2, 1]), name='weight1')
b1 = tf.Variable(tf.random.normal([2]), name='bias1')

@tf.function
def layer(X):
    return tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random.normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random.normal([1]), name='bias2')

@tf.function
def hypothesis(X):
    layer1 = layer(X)
    return tf.sigmoid(tf.matmul(layer1, W2) + b2)

@tf.function
def cost(X, Y):
    hypo = hypothesis(X)
    return -tf.reduce_mean(Y * tf.math.log(hypo) + (1-Y) * tf.math.log(1 - hypo))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

@tf.function
def predict(X):
    hypo = hypothesis(X)
    return tf.cast(hypo > 0.5, dtype=tf.float32)

@tf.function
def accuracy(X, Y):
    pred = predict(X)
    return tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

X_val = tf.Variable(x_data, dtype=tf.float32)
Y_val = tf.Variable(y_data, dtype=tf.float32)

for step in range(10001):
    with tf.GradientTape() as tp:
        cost_val = cost(X_val, Y_val)

    train = optimizer.minimize(cost_val, var_list=[W1, W2, b1, b2], tape=tp)
    if step % 1000 == 0:
        print(f'step = {step:06}, cost : {cost_val:.10}, '
              f'weight= {tf.squeeze(W1.numpy())}, '
              f'{tf.squeeze(W2.numpy())}')

h = hypothesis(X_val)
p = predict(X_val)
a = accuracy(X_val, Y_val)
print(f'Hypothesis : {tf.squeeze(h).numpy()}')
print(f'Correct : {tf.squeeze(p).numpy()}')
print(f'Accuracy : {a*100:.6}%')
```

- 결과

```cmd
step = 000000, cost : 1.251919985, weight= [-1.1580251   0.56468105], [-1.0531659 -0.0031312]
step = 001000, cost : 0.03490746021, weight= [-6.3111134  6.225426 ], [-7.4134145  7.888162 ]
step = 002000, cost : 0.01232860424, weight= [-7.095383  7.04305 ], [-9.386225  9.850936]
step = 003000, cost : 0.007414967753, weight= [-7.4342546  7.3925147], [-10.354986  10.814405]
step = 004000, cost : 0.005289349705, weight= [-7.6462774  7.6101604], [-11.000792  11.457028]
step = 005000, cost : 0.004106897861, weight= [-7.798885  7.766379], [-11.485647  11.9397  ]
step = 006000, cost : 0.003354842309, weight= [-7.917298   7.8873634], [-11.873903  12.326333]
step = 007000, cost : 0.002834767103, weight= [-8.013612  7.98563 ], [-12.197714  12.648865]
step = 008000, cost : 0.002453850117, weight= [-8.094529  8.0681  ], [-12.475452  12.925566]
step = 009000, cost : 0.002162836026, weight= [-8.164137  8.138969], [-12.718604  13.167844]
step = 010000, cost : 0.001933408435, weight= [-8.225096  8.200988], [-12.934839  13.383337]
Hypothesis : [0.00172363 0.99828726 0.997436   0.00172469]
Correct : [0. 1. 1. 0.]
Accuracy : 100.0%
```

### Neural Network에서 neural(=weight)를 늘리려면?

- weight 변수 저장 시 , shape 변형하면 됨