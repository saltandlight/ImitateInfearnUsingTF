import tensorflow as tf
import numpy as np

# 동물 학습 데이터 불러옴
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# 0<= Y <= 6
nb_classes = 7

# x1, x2, x3, ..., x16
X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 16])
Y = tf.Variable(y_data, dtype=tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random.normal([16, nb_classes], name='weight'))
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')


@tf.function
def logits():
    return tf.matmul(X, W) + b

@tf.function
def hypo(lg):
    return tf.nn.softmax(lg)

@tf.function
def cost(lg): #
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=lg, labels=Y_one_hot)
    return tf.reduce_mean(cost_i)

@tf.function
def predict(lg):
    hyp = hypo(lg)
    return tf.math.argmax(hyp, 1)

@tf.function
def accuracy(lg):
    hyp = hypo(lg)
    pred = predict(hyp)
    cost_i = tf.equal(pred, tf.math.argmax(Y_one_hot, 1))
    return tf.reduce_mean(tf.cast(cost_i, tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step in range(2000):
    with tf.GradientTape() as tp:
        lg = logits()
        loss = cost(lg)
        acc = accuracy(lg)

    train = optimizer.minimize(loss, var_list=[W, b], tape=tp)

    if step % 100 == 0:
        print(f'STEP = {step:05}, Loss = {loss:.3f}, 정확도 = {acc:.2%}')

pred = predict(logits())

for p, y in zip(pred, y_data.flatten()):
    print(f'예측여부 : [{p == int(y)}], 예측값={p:2}, 실제 Y값={int(y):2}')