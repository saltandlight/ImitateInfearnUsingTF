# Lession2 연습: Linear Regression의 Hypothesis와 cost개념

## 1. TensorFlow의 연산 이용하여 그래프 만들기

- 예측값에 대한 함수 H(x) = Wx + b를 TensorFlow로 작성

- TensorFlow에서는 weight W 와 bias b 를 변수(variable) node로 정의

  - TensorFlow가 사용하는 변수

    - 프로그램 실행하면 TensorFlow가 자체적으로 변경시키는 값을 의미
    - 학습 위해 TensorFlow가 변경하면서 최적의 값 찾아냄

  - **TensorFlow 변수 만들 때, 변수의 shape를 정의하고 값을 줘야 함**

    - W, b 값 모르니까 random한 값 줌
      - `tf.random_normal(shape)`

  - 예측값 함수를 TensorFlow의 한 점(노드)로 정의

    - tensor 또는 node W와 b로 표현

    - ```python
      import tensorflow as tf
      
      # 학습용 X와 Y 데이터를 줌
      x_train = [1, 2, 3]
      y_train = [1, 2, 3]
      
      # TensorFlow 변수를 1차원 배열로 정의
      w = tf.Variable(tf.random.normal([1]), name='weight')
      b = tf.Variable(tf.random.normal([1]), name='bias')
      
      # 예측값 함수
      hypothesis = w * x_train + b
      ```

  - cost(loss)를 TensorFlow 함수 이용해서 표현

    - tf.reduce_mean(): tensor가 주어졌을 때 평균 구하는 함수

    - cost(loss) 함수의 평균 내는 부분 구현하는 것

    - ``` cmd
      >>> import tensorflow as tf
      >>> t = [1., 2., 3., 4.]
      >>> r_mean = tf.reduce_mean(t)
      2021-12-14 11:23:46.026105: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
      To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
      2021-12-14 11:23:46.400409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
      >>> print(r_mean.numpy())
      2.5
      ```

  - cost 함수는 다음과 같이 작성 가능

    - ```
      # cost function
      @tf.function
      def cost():
          error = tf.reduce_mean(tf.square(hypothesis - y_train))
          return error
      ```

  - cost 함수를 최소화해야 함

    - TensorFlow에서는 GradientDescentOptimizer() (tf.keras.optimizers.SGD)함수를 사용

    - 학습 최적화 위해  minimize() 함수 사용해 cost function의 최솟값 찾도록 함

    - ```python
      # GradientDescentOptimizer() 함수로 학습에 Gradient Descent 최적화 방법 사용
      # v1에서는 GradientDescentOptimizer였지만 v2에서는 tf.keras.oprimizers.SGD
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      
      cost_value = cost()
      train = tf.keras.optimizers.Adam().minimize(cost_value, var_list=[w, b], tape=tf.GradientTape())
      ```

  - training 점: cost 함수를 최소화시킬 수 있음

    - train을 실행 -> cost 최소화 가능 -> 예측값과 실제값의 차이를 최소화

    - ```python
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
      
      ```

    - 변수들 업데이트

      - ```cmd
        Step = 0, cost = 0.12814129889011383, weight = [0.8066781], bias = [0.06875874]
        Step = 2000, cost = 2.1083598767290823e-05, weight = [0.9954421], bias = [0.01141209]
        Step = 4000, cost = 1.68885071616387e-05, weight = [0.9960663], bias = [0.01005096]
        Step = 6000, cost = 1.3843149645254016e-05, weight = [0.9965772], bias = [0.00894657]
        Step = 8000, cost = 1.1502769666549284e-05, weight = [0.99701554], bias = [0.00800719]
        Step = 10000, cost = 9.697023415355943e-06, weight = [0.99739516], bias = [0.00719024]
        Step = 12000, cost = 8.222054930229206e-06, weight = [0.99773663], bias = [0.00646469]
        Step = 14000, cost = 7.262739018187858e-06, weight = [0.9980055], bias = [0.00585283]
        Step = 16000, cost = 6.425103038054658e-06, weight = [0.9982439], bias = [0.00534288]
        Step = 18000, cost = 5.677965873474022e-06, weight = [0.99847186], bias = [0.00486343]
        Step = 20000, cost = 5.200541181693552e-06, weight = [0.9986533], bias = [0.00445441]
        ```

      - 예측 결과: W=1, b=0

    - cost(loss) 함수의 그래프 그리기

      - ```python
        import tensorflow as tf
        import matplotlib.pyplot as plt
        
        X = [1, 2, 3]
        Y = [1, 2, 3]
        
        @tf.function
        def hypothesis(W):
            return tf.multiply(W, X)
        
        @tf.function
        def cost(W):
            return tf.reduce_mean(tf.square(hypothesis(W) - Y))
        
        W_val = []
        cost_val = []
        
        for i in range(-30, 50):
            feed_W = i * 0.1
            curr_W = tf.Variable(feed_W, dtype=tf.float32)
        
            curr_cost = cost(feed_W)
        
            W_val.append(curr_W)
            cost_val.append(curr_cost)
        
        plt.plot(W_val, cost_val)
        plt.show()
        ```

      - <img src="pic\Figure_1.png" style="zoom:67%;" />

  - cost(loss) 함수의 극소값 찾기

    - ```python
      import tensorflow as tf
      
      x_data = [1, 2, 3]
      y_data = [1, 2, 3]
      
      W = tf.Variable(tf.random.normal([1]), name='weight')
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
          gradient = tf.reduce_mean(tf.multiply((hypothesis() - Y) , X))
          descent = W - learning_rate * gradient
          return W.assign(descent)
      
      for step in range(21):
          update_rs = update()
          cost_rs = cost()
          print(f"STEP = {step:02}, cost(loss) 함수값 = {cost_rs},"
                f"기울기 = {W.numpy()}")
      ```

    - ```cmd
      STEP = 00, cost(loss) 함수값 = 0.6926255226135254,기울기 = [0.61474717]
      STEP = 01, cost(loss) 함수값 = 0.19701345264911652,기울기 = [0.7945318]
      STEP = 02, cost(loss) 함수값 = 0.05603940412402153,기울기 = [0.890417]
      STEP = 03, cost(loss) 함수값 = 0.01594008132815361,기울기 = [0.94155574]
      STEP = 04, cost(loss) 함수값 = 0.004534053150564432,기울기 = [0.96882975]
      STEP = 05, cost(loss) 함수값 = 0.0012896895641461015,기울기 = [0.98337585]
      STEP = 06, cost(loss) 함수값 = 0.00036684147198684514,기울기 = [0.9911338]
      STEP = 07, cost(loss) 함수값 = 0.00010434724390506744,기울기 = [0.9952713]
      STEP = 08, cost(loss) 함수값 = 2.968206536024809e-05,기울기 = [0.997478]
      STEP = 09, cost(loss) 함수값 = 8.442278158327099e-06,기울기 = [0.99865496]
      STEP = 10, cost(loss) 함수값 = 2.4012849735299824e-06,기울기 = [0.99928266]
      STEP = 11, cost(loss) 함수값 = 6.830818506387004e-07,기울기 = [0.9996174]
      STEP = 12, cost(loss) 함수값 = 1.9437236176145234e-07,기울기 = [0.9997959]
      STEP = 13, cost(loss) 함수값 = 5.5306017543443886e-08,기울기 = [0.99989116]
      STEP = 14, cost(loss) 함수값 = 1.5714576306891104e-08,기울기 = [0.99994195]
      STEP = 15, cost(loss) 함수값 = 4.483050819459322e-09,기울기 = [0.999969]
      STEP = 16, cost(loss) 함수값 = 1.274084837632472e-09,기울기 = [0.9999835]
      STEP = 17, cost(loss) 함수값 = 3.6315364604355693e-10,기울기 = [0.9999912]
      STEP = 18, cost(loss) 함수값 = 1.0291145713381411e-10,기울기 = [0.9999953]
      STEP = 19, cost(loss) 함수값 = 2.9847530697013624e-11,기울기 = [0.9999975]
      STEP = 20, cost(loss) 함수값 = 7.716494110354688e-12,기울기 = [0.9999987]
      
      Process finished with exit code 0
      
      ```

참고: `https://forensics.tistory.com/7?category=767888`

`https://stackoverflow.com/questions/68879963/valueerror-tape-is-required-when-a-tensor-loss-is-passed`

