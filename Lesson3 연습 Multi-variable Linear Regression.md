# Lession 3 연습: Multi-variable Linear Regression) 연습

- H(x1, x2, x3) = x1 * w1 + x2 * w2 + x3 * w3 + b

- 변수가 1개일 때 구현했던 코드 방식으로 구현

  - ```python
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
    
    @tf.function
    def hypothesis():
        return tf.multiply(x1, w1) + tf.multiply(x2, w2) + tf.multiply(x3, w3) + b
    ```

- 텐서플로우 그래프로 구현을 위한 전체 코드

  - ```python
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
    
    for step in range(2001):
        with tf.GradientTape() as tp:
            hy_val, cost_val = cost(trainable_vars)
    
        train = tf.keras.optimizers.Adam().minimize(cost_val, var_list=trainable_vars, tape=tp)
    
        if step % 10 == 0:
            print(f"STEP = {step:04}, cost = {cost_val}, 예측값: {hy_val}")
    ```

- 변수가 3개 데이터를 행렬식을 이용하여 텐서플로우로 구현하기

  - ```python
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
    ```

  - 실행결과

    ```cmd
    STEP = 000000, cost = 65706.8359375, 예측값: [[-75.959076]
     [-88.97388 ]
     [-88.94004 ]
     [-94.885376]
     [-69.102165]]
    STEP = 005000, cost = 1.65382981300354, 예측값: [[150.73233]
     [184.93634]
     [180.22174]
     [197.993  ]
     [140.37614]]
    STEP = 010000, cost = 1.3054295778274536, 예측값: [[150.62599]
     [185.03629]
     [180.22348]
     [197.72705]
     [140.733  ]]
    STEP = 015000, cost = 1.063015103340149, 예측값: [[150.67543]
     [185.02632]
     [180.26897]
     [197.52339]
     [140.9198 ]]
    STEP = 020000, cost = 0.8748529553413391, 예측값: [[150.75632]
     [184.99208]
     [180.32062]
     [197.35123]
     [141.05193]]
    ```

- 파일에서 불러와 데이터를 학습시키기

  - ```python
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
    
    ```

  - ```cmd
    D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercisetf7.py
    2021-12-15 14:15:32.956832: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2021-12-15 14:15:33.341935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
    WARNING:tensorflow:AutoGraph could not transform <function cost at 0x000001FC3A285EE8> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: 'arguments' object has no attribute 'posonlyargs'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING:tensorflow:AutoGraph could not transform <function hypothesis at 0x000001FC3A41D438> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module 'gast' has no attribute 'Constant'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    STEP = 0, cost = 12908.5791015625, hy_val = [[48.8714  ]
     [53.57257 ]
     [55.43933 ]
     [62.326828]
     [37.63072 ]
     [29.48714 ]
     [50.493706]
     [43.10822 ]
     [50.906353]
     [50.549355]
     [45.096508]
     [42.720055]
     [55.368156]
     [42.7937  ]
     [50.11467 ]
     [56.621758]
     [36.776833]
     [63.326893]
     [53.26853 ]
     [48.421177]
     [58.050003]
     [52.2841  ]
     [55.220753]
     [44.780773]
     [55.580326]]
    2021-12-15 14:15:34.184953: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
    STEP = 100, cost = 13.263219833374023, hy_val = [[155.5599 ]
     [181.97658]
     [181.86858]
     [200.01208]
     [135.60776]
     [101.73292]
     [153.49832]
     [119.38389]
     [170.94232]
     [161.73456]
     [144.39404]
     [140.70416]
     [185.99625]
     [151.40411]
     [153.61813]
     [186.89993]
     [140.27519]
     [186.89842]
     [177.77783]
     [159.80739]
     [178.86636]
     [172.72736]
     [170.39655]
     [152.36835]
     [188.63174]]
    STEP = 200, cost = 12.889181137084961, hy_val = [[155.48067 ]
     [182.05026 ]
     [181.85463 ]
     [199.99109 ]
     [135.71068 ]
     [101.819496]
     [153.42648 ]
     [119.2472  ]
     [171.04324 ]
     [161.83025 ]
     [144.38118 ]
     [140.76373 ]
     [185.99095 ]
     [151.41956 ]
     [153.57054 ]
     [186.9535  ]
     [140.34944 ]
     [186.76938 ]
     [177.7477  ]
     [159.76103 ]
     [178.8085  ]
     [172.77992 ]
     [170.32872 ]
     [152.30833 ]
     [188.68443 ]]
    STEP = 300, cost = 12.534626007080078, hy_val = [[155.40388 ]
     [182.12193 ]
     [181.84125 ]
     [199.97064 ]
     [135.8109  ]
     [101.903366]
     [153.35632 ]
     [119.11365 ]
     [171.14099 ]
     [161.92233 ]
     [144.3685  ]
     [140.8213  ]
     [185.98624 ]
     [151.4353  ]
     [153.52388 ]
     [187.0054  ]
     [140.42255 ]
     [186.64343 ]
     [177.7189  ]
     [159.71643 ]
     [178.75189 ]
     [172.83081 ]
     [170.26254 ]
     [152.25089 ]
     [188.73587 ]]
    STEP = 400, cost = 12.198429107666016, hy_val = [[155.32951 ]
     [182.19165 ]
     [181.82846 ]
     [199.95073 ]
     [135.90852 ]
     [101.9846  ]
     [153.28777 ]
     [118.983154]
     [171.2357  ]
     [162.01097 ]
     [144.35603 ]
     [140.8769  ]
     [185.98213 ]
     [151.45132 ]
     [153.4781  ]
     [187.0557  ]
     [140.49454 ]
     [186.5205  ]
     [177.69145 ]
     [159.67351 ]
     [178.6965  ]
     [172.8801  ]
     [170.19798 ]
     [152.19585 ]
     [188.78609 ]]
    STEP = 500, cost = 11.879690170288086, hy_val = [[155.2574  ]
     [182.25943 ]
     [181.81616 ]
     [199.9313  ]
     [136.00357 ]
     [102.06326 ]
     [153.22078 ]
     [118.855644]
     [171.32741 ]
     [162.0963  ]
     [144.3438  ]
     [140.9306  ]
     [185.97853 ]
     [151.46758 ]
     [153.4332  ]
     [187.10437 ]
     [140.56535 ]
     [186.40048 ]
     [177.66519 ]
     [159.63219 ]
     [178.64232 ]
     [172.9278  ]
     [170.13498 ]
     [152.14313 ]
     [188.83505 ]]
    STEP = 600, cost = 11.577422142028809, hy_val = [[155.18752]
     [182.32536]
     [181.8044 ]
     [199.91237]
     [136.09612]
     [102.13948]
     [153.15533]
     [118.73103]
     [171.41628]
     [162.17839]
     [144.33173]
     [140.9825 ]
     [185.97546]
     [151.48404]
     [153.38918]
     [187.15154]
     [140.63506]
     [186.28334]
     [177.64017]
     [159.59242]
     [178.58932]
     [172.97398]
     [170.0735 ]
     [152.09267]
     [188.88286]]
    STEP = 700, cost = 11.290752410888672, hy_val = [[155.11981]
     [182.38953]
     [181.79315]
     [199.89395]
     [136.18626]
     [102.21333]
     [153.09143]
     [118.60933]
     [171.5024 ]
     [162.25746]
     [144.3199 ]
     [141.03265]
     [185.9729 ]
     [151.50069]
     [153.34602]
     [187.19727]
     [140.70364]
     [186.169  ]
     [177.6163 ]
     [159.55414]
     [178.53752]
     [173.01874]
     [170.01355]
     [152.04437]
     [188.92953]]
    STEP = 800, cost = 11.018837928771973, hy_val = [[155.05417 ]
     [182.45189 ]
     [181.78232 ]
     [199.87595 ]
     [136.27405 ]
     [102.284874]
     [153.02899 ]
     [118.49041 ]
     [171.58582 ]
     [162.33356 ]
     [144.30826 ]
     [141.08109 ]
     [185.97072 ]
     [151.51746 ]
     [153.3037  ]
     [187.24152 ]
     [140.77109 ]
     [186.05737 ]
     [177.59346 ]
     [159.51727 ]
     [178.48682 ]
     [173.06204 ]
     [169.95502 ]
     [151.99805 ]
     [188.97502 ]]
    STEP = 900, cost = 10.760908126831055, hy_val = [[154.99055]
     [182.5126 ]
     [181.77199]
     [199.85843]
     [136.35956]
     [102.3542 ]
     [152.96802]
     [118.37423]
     [171.66669]
     [162.40683]
     [144.29683]
     [141.12793]
     [185.96902]
     [151.53436]
     [153.26222]
     [187.28442]
     [140.8374 ]
     [185.94846]
     [177.57173]
     [159.48181]
     [178.43726]
     [173.10403]
     [169.89795]
     [151.95375]
     [189.01947]]
    STEP = 1000, cost = 10.516196250915527, hy_val = [[154.92885]
     [182.57161]
     [181.76204]
     [199.84132]
     [136.44281]
     [102.42138]
     [152.90842]
     [118.26072]
     [171.74501]
     [162.47736]
     [144.28557]
     [141.17314]
     [185.96767]
     [151.55135]
     [153.22154]
     [187.32599]
     [140.9026 ]
     [185.84209]
     [177.55092]
     [159.44762]
     [178.38878]
     [173.14467]
     [169.84221]
     [151.91129]
     [189.06277]]
    STEP = 1100, cost = 10.284049987792969, hy_val = [[154.86906 ]
     [182.62907 ]
     [181.75255 ]
     [199.82472 ]
     [136.52393 ]
     [102.486496]
     [152.85023 ]
     [118.149864]
     [171.82097 ]
     [162.54532 ]
     [144.27455 ]
     [141.2169  ]
     [185.96675 ]
     [151.56844 ]
     [153.18172 ]
     [187.36629 ]
     [140.96669 ]
     [185.73833 ]
     [177.53114 ]
     [159.41472 ]
     [178.34138 ]
     [173.18407 ]
     [169.78789 ]
     [151.87065 ]
     [189.10509 ]]
    STEP = 1200, cost = 10.06372356414795, hy_val = [[154.81107]
     [182.68497]
     [181.74342]
     [199.80849]
     [136.60292]
     [102.54961]
     [152.7934 ]
     [118.04155]
     [171.89459]
     [162.61073]
     [144.2637 ]
     [141.25919]
     [185.96614]
     [151.58553]
     [153.14267]
     [187.40535]
     [141.02966]
     [185.63702]
     [177.51222]
     [159.38304]
     [178.29504]
     [173.22221]
     [169.73486]
     [151.8317 ]
     [189.14636]]
    STEP = 1300, cost = 9.854666709899902, hy_val = [[154.75482 ]
     [182.73932 ]
     [181.73466 ]
     [199.79265 ]
     [136.67986 ]
     [102.61076 ]
     [152.73788 ]
     [117.935745]
     [171.96594 ]
     [162.67374 ]
     [144.25305 ]
     [141.30005 ]
     [185.96584 ]
     [151.60265 ]
     [153.10442 ]
     [187.44319 ]
     [141.09152 ]
     [185.53815 ]
     [177.49417 ]
     [159.35252 ]
     [178.24971 ]
     [173.25917 ]
     [169.68312 ]
     [151.79439 ]
     [189.18661 ]]
    STEP = 1400, cost = 9.65624713897705, hy_val = [[154.7003  ]
     [182.79225 ]
     [181.72632 ]
     [199.77727 ]
     [136.7548  ]
     [102.670074]
     [152.68367 ]
     [117.83241 ]
     [172.03516 ]
     [162.73445 ]
     [144.24261 ]
     [141.33958 ]
     [185.96587 ]
     [151.6198  ]
     [153.06694 ]
     [187.4799  ]
     [141.15231 ]
     [185.44168 ]
     [177.47697 ]
     [159.32314 ]
     [178.20546 ]
     [173.29503 ]
     [169.63268 ]
     [151.75868 ]
     [189.22594 ]]
    STEP = 1500, cost = 9.467931747436523, hy_val = [[154.64742]
     [182.84372]
     [181.71828]
     [199.76222]
     [136.8278 ]
     [102.72756]
     [152.63074]
     [117.73147]
     [172.10222]
     [162.79291]
     [144.23235]
     [141.37779]
     [185.96617]
     [151.63689]
     [153.03023]
     [187.51547]
     [141.21199]
     [185.34749]
     [177.46056]
     [159.29482]
     [178.16214]
     [173.32976]
     [169.58345]
     [151.72447]
     [189.26427]]
    STEP = 1600, cost = 9.289194107055664, hy_val = [[154.59612]
     [182.89381]
     [181.71057]
     [199.74759]
     [136.89891]
     [102.78328]
     [152.57901]
     [117.63287]
     [172.16725]
     [162.84923]
     [144.22227]
     [141.41475]
     [185.96672]
     [151.65398]
     [152.99426]
     [187.54994]
     [141.27058]
     [185.25557]
     [177.44487]
     [159.26753]
     [178.1198 ]
     [173.36339]
     [169.5354 ]
     [151.6917 ]
     [189.30167]]
    STEP = 1700, cost = 9.11950969696045, hy_val = [[154.54636 ]
     [182.94255 ]
     [181.70319 ]
     [199.7333  ]
     [136.96817 ]
     [102.837326]
     [152.5285  ]
     [117.53659 ]
     [172.23032 ]
     [162.9035  ]
     [144.2124  ]
     [141.45047 ]
     [185.96753 ]
     [151.67099 ]
     [152.95905 ]
     [187.5834  ]
     [141.32811 ]
     [185.16586 ]
     [177.42992 ]
     [159.24124 ]
     [178.0784  ]
     [173.396   ]
     [169.48853 ]
     [151.66028 ]
     [189.33818 ]]
    STEP = 1800, cost = 8.95842170715332, hy_val = [[154.49808 ]
     [182.98999 ]
     [181.69614 ]
     [199.71942 ]
     [137.03564 ]
     [102.889725]
     [152.47919 ]
     [117.442535]
     [172.29147 ]
     [162.9558  ]
     [144.20271 ]
     [141.48505 ]
     [185.96858 ]
     [151.68796 ]
     [152.92456 ]
     [187.61584 ]
     [141.3846  ]
     [185.07831 ]
     [177.41566 ]
     [159.21591 ]
     [178.03796 ]
     [173.42763 ]
     [169.44281 ]
     [151.63022 ]
     [189.37381 ]]
    STEP = 1900, cost = 8.805481910705566, hy_val = [[154.45126]
     [183.03616]
     [181.68933]
     [199.70584]
     [137.10138]
     [102.94054]
     [152.43103]
     [117.35068]
     [172.3508 ]
     [163.00618]
     [144.1932 ]
     [141.51848]
     [185.96983]
     [151.70485]
     [152.8908 ]
     [187.64728]
     [141.44002]
     [184.99289]
     [177.40207]
     [159.19151]
     [177.99844]
     [173.45828]
     [169.39821]
     [151.60141]
     [189.40857]]
    STEP = 2000, cost = 8.660269737243652, hy_val = [[154.40584]
     [183.0811 ]
     [181.68285]
     [199.69266]
     [137.16542]
     [102.98984]
     [152.38402]
     [117.26099]
     [172.40834]
     [163.05475]
     [144.18388]
     [141.55083]
     [185.97128]
     [151.72166]
     [152.85773]
     [187.6778 ]
     [141.49442]
     [184.90952]
     [177.3891 ]
     [159.16801]
     [177.95981]
     [173.488  ]
     [169.3547 ]
     [151.57382]
     [189.44249]]
    예상 점수는 [[179.12897]]점입니다.
    예상 점수는 [[191.50174]
     [175.91647]]입니다.
    
    Process finished with exit code 0
    
    ```

- 데이터가 많은 경우 Queue Runners를 이용하기

  - 일단.. Skip..!

