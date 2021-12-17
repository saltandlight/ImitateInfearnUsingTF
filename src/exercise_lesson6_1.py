import tensorflow as tf

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]

y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

W = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.random.normal([3]))

@tf.function
def hypo(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

@tf.function
def cost(X, Y):
    hy_val = hypo(X)
    return tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hy_val), axis=1))

@tf.function
def predict(X):
    hy_val = hypo(X)
    return tf.math.argmax(hy_val, 1)

@tf.function
def accuracy(predict, Y):
    is_correct = tf.equal(predict, tf.math.argmax(Y, 1))

    return tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-15)

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 3])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 3])

X2 = tf.Variable(x_test, dtype=tf.float32, shape=[None, 3])
Y2 = tf.Variable(y_test, dtype=tf.float32, shape=[None, 3])

# 모델의 성능 평가
for step in range(20):
    with tf.GradientTape() as tp:
        cost_val = cost(X, Y)

    train = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    print(f'STEP = {step:>03}, cost_val = {cost_val}, \n\tW_val = {W.numpy()}')

# 테스트 데이터로 결과 예측

pd = predict(X2)
print(f'Prediction: {pd}')
print(f'Accuracy: {accuracy(pd, Y2)}')