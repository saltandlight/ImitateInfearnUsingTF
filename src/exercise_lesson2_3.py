import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W = tf.Variable(tf.random.normal([1]), name='weight')
W = tf.Variable(5, dtype=tf.float32, name='weight')
X = tf.Variable(x_data, dtype=tf.float32)
Y = tf.Variable(y_data, dtype=tf.float32)

@tf.function
def hypothesis():
    return tf.multiply(W, X)

@tf.function
def cost():
    hypo = hypothesis()
    return tf.reduce_mean(tf.square(hypo - Y))

# W = W - learning_rate * 미분계수
@tf.function
def update():
    learning_rate = 0.1
    gradient = tf.reduce_mean(tf.multiply((hypothesis() - Y), X))
    descent = W - learning_rate * gradient
    return W.assign(descent)

for step in range(21):
    update_rs = update()
    cost_rs = cost()
    print(f"STEP = {step:03}, cost(loss) 함수값 = {cost_rs},"
          f"기울기 = {W.numpy()}")

