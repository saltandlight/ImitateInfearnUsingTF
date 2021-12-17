import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# X, Y는 이따가 한꺼번에 정리
X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 2])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([2, 1]), dtype=tf.float32, name='weight', shape=[2, 1])
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias', shape=[1])

@tf.function
def hypothesis():
    return tf.sigmoid(tf.matmul(X, W)) + b

@tf.function
def cost(ht):
    return -tf.reduce_mean(Y*tf.math.log(ht) + (1-Y)*tf.math.log(1-ht))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
trainable_vars = [W, b]
print(f"trainable_vars={trainable_vars}")

@tf.function
def predict(ht):
    return tf.cast(ht > 0.5, dtype=tf.float32)

@tf.function
def accuracy(pt):
    return tf.reduce_mean(tf.cast(tf.equal(pt, Y), dtype=tf.float32))

for step in range(10001):
    with tf.GradientTape() as tp:
        ht_val = hypothesis()
        cost_val = cost(abs(ht_val))

    train_val = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    if step % 2000 == 0:
        print(f"STEP = {step:06}, cost 함수값 = {cost_val}, 가설의 값 = {ht_val}")

h = hypothesis()
c = predict(h)
a = accuracy(c)
print(f"가설식의 값 = {h},\n실제의 값 = {c},\n정확도 = {a}")
