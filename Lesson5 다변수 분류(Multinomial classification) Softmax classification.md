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

