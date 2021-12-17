import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5],
          [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]

y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 4])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 3])

# Y값의 갯수 = 분류대상의 개수
nb_classes = 3

W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

# 가설식은 softmax 함수 사용하여 표현
@tf.function
def hypo(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

# Cost 함수는 cross-entropy 사용하여 표현
@tf.function
def cost(hypo):
    return tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypo), axis=1))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step in range(2001):
    with tf.GradientTape() as tp:
        ht_val = hypo(X)
        cost_val = cost(abs(ht_val))

    train_val = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)

    if step % 200 == 0:
        print(f"STEP = {step:04}, cost 함수값 = {cost_val}")

# ONE-HOT Encoding Test
a = hypo(tf.Variable([[1, 11, 7, 9]], dtype=tf.float32))
print(f"예측 결과 = {a}, ONE-HOT 인코딩 결과 = {tf.math.argmax(a, 1)}")

b = hypo(tf.Variable([[1, 3, 4, 3]], dtype=tf.float32))
print(f"예측 결과 = {b}, ONE-HOT 인코딩 결과 = {tf.math.argmax(b, 1)}")

c = hypo(tf.Variable([[1, 1, 0, 1]], dtype=tf.float32))
print(f"예측 결과 = {c}, ONE-HOT 인코딩 결과 = {tf.math.argmax(c, 1)}")

# 한번에 여러 값을 넣어 Test
all = hypo(tf.Variable([[1, 11, 7, 9], [1, 3, 4, 3],[1, 1, 0, 1]], dtype=tf.float32))
print(f"예측 결과 = {all}, ONE-HOT 인코딩 결과 = {tf.math.argmax(all)}")