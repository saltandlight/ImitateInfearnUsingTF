import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.],
          [96., 98., 100.], [73., 66., 70]]

y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 3])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 1])

# weight과 bias 는 난수 발생-> 조정하는 값=> 변수로 설정
W = tf.Variable(tf.random.normal([3, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 가설식 함수
@tf.function
def hypothesis(vars):
    return tf.matmul(X, vars[0]) + vars[1]

# cost(loss) 함수
@tf.function
def cost(vars):
    ht = hypothesis(vars)
    return ht, tf.reduce_mean(tf.square(ht - Y))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
trainable_vars = [W, b]

for step in range(20001):
    with tf.GradientTape() as tp:
        hy_val, cost_val = cost(trainable_vars)

    train = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)

    if step % 5000 == 0:
        print(f"STEP = {step:06}, cost = {cost_val}, 예측값: {hy_val}")