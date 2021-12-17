import tensorflow as tf

# 학습용 X와 Y 데이터를 줌
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# TensorFlow 변수를 1차원 배열로 정의
w = tf.Variable(tf.random.normal([1]), trainable=True, name='weight')
b = tf.Variable(tf.random.normal([1]), trainable=True, name='bias')

# cost function
@tf.function
def cost(w, b):
    # 예측값 함수
    hypothesis = w * x_train + b
    error = tf.reduce_mean(tf.square(y_train - hypothesis))
    return error

# GradientDescentOptimizer() 함수로 학습에 Gradient Descent 최적화 방법 사용
# v1에서는 GradientDescentOptimizer였지만 v2에서는 tf.keras.oprimizers.SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

trainable_vars = [w, b]

# 변수 초기화하지 않아도 Tensorflow v2에서는 변수 생성 시 초기화함

epochs = 20000
for step in range(epochs+1):
    with tf.GradientTape() as tp:
        cost_fn = cost(w, b)

    train = tf.keras.optimizers.Adam().minimize(cost_fn, var_list=trainable_vars, tape=tp)
    if step % 2000 == 0:
        print(f"Step = {step}, cost = {cost_fn}, "
              f"weight = {w.numpy()}, bias = {b.numpy()}")