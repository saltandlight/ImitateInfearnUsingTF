# Lesson5. 다변수 분류(Multinomial classification): Softmax classification

- 새로 정의할 Cost Function => Y = WX + b에 Softmax 함수를 씌워준 함수 
- H(x) = Softmax(Y) = Softmax(WX + b)  (v1 버전)
  - `hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)`

- cost function  (v1 버전)
  - `cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))`
- 경사하강법  (v1 버전)
  - `optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)` 

## Sotftmax 분류 함수를 사용하는 기계학습 연습

- y는 언제나 0~1(확률)
- 데이터 사용 시 모양이 중요
- code

```python
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
def hypo():
    return tf.nn.softmax(tf.matmul(X, W) + b)

# Cost 함수는 cross-entropy 사용하여 표현
@tf.function
def cost(hypo):
    return tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypo), axis=1))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step in range(2001):
    with tf.GradientTape() as tp:
        ht_val = hypo()
        cost_val = cost(abs(ht_val))

    train_val = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)

    if step % 200 == 0:
        print(f"STEP = {step:04}, cost 함수값 = {cost_val}")
```

- 결과

```cmd
D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercisetf10.py
2021-12-16 17:19:40.024347: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-16 17:19:40.375834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
WARNING:tensorflow:AutoGraph could not transform <function hypo at 0x0000019CAFE45EE8> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2021-12-16 17:19:41.164416: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
WARNING:tensorflow:AutoGraph could not transform <function cost at 0x0000019CAFFDDF78> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
STEP = 0000, cost 함수값 = 7.2942023277282715
STEP = 0200, cost 함수값 = 0.6680079698562622
STEP = 0400, cost 함수값 = 0.5624895095825195
STEP = 0600, cost 함수값 = 0.47354856133461
STEP = 0800, cost 함수값 = 0.3818315863609314
STEP = 1000, cost 함수값 = 0.2925211191177368
STEP = 1200, cost 함수값 = 0.23186901211738586
STEP = 1400, cost 함수값 = 0.21056905388832092
STEP = 1600, cost 함수값 = 0.19282425940036774
STEP = 1800, cost 함수값 = 0.1777079552412033
STEP = 2000, cost 함수값 = 0.16473838686943054
```

## ONE-HOT Encoding Test

- 추가한 코드

```python
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
```

- 결과

```cmd
D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercisetf11.py
2021-12-16 17:35:44.720402: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-16 17:35:45.074241: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
WARNING:tensorflow:AutoGraph could not transform <function hypo at 0x000001AB0FC95EE8> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2021-12-16 17:35:45.861098: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
WARNING:tensorflow:AutoGraph could not transform <function cost at 0x000001AB0FE2DF78> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'arguments' object has no attribute 'posonlyargs'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
STEP = 0000, cost 함수값 = 4.068014621734619
STEP = 0200, cost 함수값 = 0.5198187828063965
STEP = 0400, cost 함수값 = 0.42381662130355835
STEP = 0600, cost 함수값 = 0.34832870960235596
STEP = 0800, cost 함수값 = 0.2710357904434204
STEP = 1000, cost 함수값 = 0.23426255583763123
STEP = 1200, cost 함수값 = 0.21230670809745789
STEP = 1400, cost 함수값 = 0.19410288333892822
STEP = 1600, cost 함수값 = 0.17868834733963013
STEP = 1800, cost 함수값 = 0.1654481291770935
STEP = 2000, cost 함수값 = 0.15400435030460358
예측 결과 = [[9.5315985e-03 9.9045956e-01 8.8453962e-06]], ONE-HOT 인코딩 결과 = [1]
예측 결과 = [[0.7998347  0.18207653 0.01808883]], ONE-HOT 인코딩 결과 = [0]
예측 결과 = [[9.3596926e-09 3.3057254e-04 9.9966943e-01]], ONE-HOT 인코딩 결과 = [2]
예측 결과 = [[2.6991877e-01 7.3008120e-01 1.2788075e-08]
 [9.9410832e-01 5.8905287e-03 1.1477929e-06]
 [1.5691009e-04 1.4425260e-01 8.5559052e-01]], ONE-HOT 인코딩 결과 = [1 0 2]
```

## Fancy softmax Classifier

### logits을 사용하는 Softmax Cross-Entropy 함수

- `tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))` <- 이 식이 복잡해서 Tensorflow에서 `tf.nn.softmax_cross_entropy_with_logits()` 함수 제공
- 입력값: 
  - logits
  - one-hot 인코딩된 Y값

- ```python
  cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one__hot)
  cost = tf.reduce_mean(cost_i)
  ```

### softmax_ cross_entropy_with_logits 함수를 사용하는 모델

#### softmax_cross_entropy_with_logits 함수를 사용하는 모델의 기계 학습 연습

- code

```python
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
def accuracy(lg): #
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
```

- 결과

```cmd
D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercisetf12.py
(101, 16) (101, 1)
2021-12-17 10:46:32.471942: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-17 10:46:32.831286: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
one_hot tf.Tensor(
[[[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 0. 0. 0. 1. 0. 0.]]

 [[0. 0. 0. 0. 1. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 0. 0. 0. 1. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 0. 1. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 1. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 1. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[0. 0. 0. 0. 1. 0. 0.]]

 [[0. 0. 1. 0. 0. 0. 0.]]

 [[0. 0. 1. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 1. 0.]]

 [[1. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 1.]]

 [[0. 1. 0. 0. 0. 0. 0.]]], shape=(101, 1, 7), dtype=float32)
reshape tf.Tensor(
[[1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0. 0.]], shape=(101, 7), dtype=float32)
STEP = 00000, Loss = 7.398, 정확도 = 16.83%
STEP = 00100, Loss = 0.733, 정확도 = 77.23%
STEP = 00200, Loss = 0.449, 정확도 = 85.15%
STEP = 00300, Loss = 0.324, 정확도 = 92.08%
STEP = 00400, Loss = 0.251, 정확도 = 92.08%
STEP = 00500, Loss = 0.203, 정확도 = 95.05%
STEP = 00600, Loss = 0.171, 정확도 = 98.02%
STEP = 00700, Loss = 0.147, 정확도 = 98.02%
STEP = 00800, Loss = 0.129, 정확도 = 99.01%
STEP = 00900, Loss = 0.116, 정확도 = 100.00%
STEP = 01000, Loss = 0.105, 정확도 = 100.00%
STEP = 01100, Loss = 0.096, 정확도 = 100.00%
STEP = 01200, Loss = 0.088, 정확도 = 100.00%
STEP = 01300, Loss = 0.081, 정확도 = 100.00%
STEP = 01400, Loss = 0.076, 정확도 = 100.00%
STEP = 01500, Loss = 0.071, 정확도 = 100.00%
STEP = 01600, Loss = 0.067, 정확도 = 100.00%
STEP = 01700, Loss = 0.063, 정확도 = 100.00%
STEP = 01800, Loss = 0.060, 정확도 = 100.00%
STEP = 01900, Loss = 0.057, 정확도 = 100.00%
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 4, 실제 Y값= 4
예측여부 : [True], 예측값= 4, 실제 Y값= 4
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 4, 실제 Y값= 4
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 2, 실제 Y값= 2
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 2, 실제 Y값= 2
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 2, 실제 Y값= 2
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 4, 실제 Y값= 4
예측여부 : [True], 예측값= 2, 실제 Y값= 2
예측여부 : [True], 예측값= 2, 실제 Y값= 2
예측여부 : [True], 예측값= 3, 실제 Y값= 3
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 1, 실제 Y값= 1
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 5, 실제 Y값= 5
예측여부 : [True], 예측값= 0, 실제 Y값= 0
예측여부 : [True], 예측값= 6, 실제 Y값= 6
예측여부 : [True], 예측값= 1, 실제 Y값= 1
```

