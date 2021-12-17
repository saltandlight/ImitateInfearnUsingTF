import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 8])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([8, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

@tf.function
def hypo():
    return tf.sigmoid(tf.matmul(X, W) + b)

@tf.function
def cost(hypo):
    return -tf.reduce_mean(Y*tf.math.log(hypo) + (1-Y) * tf.math.log(1-hypo))

@tf.function
def predict(hypo):
    return tf.cast(hypo > 0.5, dtype=tf.float32)

@tf.function
def accuracy(pd):
    return tf.reduce_mean(tf.cast(tf.equal(pd, Y), dtype=tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for step in range(10001):
    with tf.GradientTape() as tp:
        ht_val = hypo()
        cost_val = cost(abs(ht_val))

    train_val = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    if step % 2000 == 0:
        print(f"STEP={step:06}, cost 함수값 = {cost_val}")

h = hypo()
p = predict(h)
a = accuracy(p)
print(f"\n 가설식의 값 = {h},\n\n실제의 값={p},\n\n정확도={a}")