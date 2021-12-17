import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float32)

# 모든 데이터 행과 처음부터 마지막 -1개의 열까지 선택
x_data = xy[:, 0: -1]
# 마지막 열만 선택
y_data = xy[:, [-1]]

# 들어온 데이터 정상적으로 들어온 건지 확인
# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 3])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([3, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 가설식 함수
@tf.function
def hypothesis(X, vars):
    return tf.matmul(X, vars[0]) + vars[1]

# cost(loss) 함수
@tf.function
def cost(vars):
    ht = hypothesis(X, vars)
    return ht, tf.reduce_mean(tf.square(ht - Y))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
trainable_vars = [W, b]
for step in range(2001):
    with tf.GradientTape() as tp:
        hy_val, cost_val = cost(trainable_vars)

    train = optimizer.minimize(cost_val, var_list=trainable_vars, tape=tp)
    if step % 100 == 0:
        print(f"STEP = {step}, cost = {cost_val}, hy_val = {hy_val}")

X2 = tf.Variable([[100, 70, 101]], dtype=tf.float32, shape=[None, 3])
print(f"예상 점수는 {hypothesis(X2, trainable_vars).numpy()}점입니다.")
X3 = tf.Variable([[60, 70, 110], [90, 100, 80]], dtype=tf.float32, shape=[None, 3])
print(f"예상 점수는 {hypothesis(X3, trainable_vars).numpy()}입니다.")
