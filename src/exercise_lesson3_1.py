import tensorflow as tf

# 훈련용 데이터 입력
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# 텐서 자료 입력(v1과는 달리 처음부터 생성함)
x1 = tf.Variable(x1_data, dtype=tf.float32)
x2 = tf.Variable(x2_data, dtype=tf.float32)
x3 = tf.Variable(x3_data, dtype=tf.float32)

Y = tf.Variable(y_data, dtype=tf.float32)


# weight과 bias는 난수 발생하여 조정 하는 값-> 변수로 설정, 모양은 1차원 벡터
w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.Variable(tf.random.normal([1]), name='weight3')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 가설식 함수
@tf.function
def hypothesis(vars):
    return tf.multiply(x1, vars[0]) + tf.multiply(x2, vars[1]) + tf.multiply(x3, vars[2]) + vars[3]

# cost(loss) 함수
@tf.function
def cost(vars):
    ht = hypothesis(vars)
    return ht, tf.reduce_mean(tf.square(ht - Y))

# 데이터에 대해 cost 함수의 극소값 구하기 위해 경사하강법 최적화 메서드 사용
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
trainable_vars = [w1, w2, w3, b]
print(w1)

for step in range(2001):
    with tf.GradientTape() as tp:
        hy_val, cost_val = cost(trainable_vars)

    train = tf.keras.optimizers.Adam().minimize(cost_val, var_list=trainable_vars, tape=tp)

    if step % 10 == 0:
        print(f"STEP = {step:04}, cost = {cost_val}, 예측값: {hy_val}")