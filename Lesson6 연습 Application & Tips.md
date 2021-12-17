# Lesson 6 연습: Application & Tips

## Learning rate와 평가 방법 연습

### `Training dataset과 test dataset 연습`

- 프로그램 코드

```python
import tensorflow as tf

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]

y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

W = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.random.normal([3]))

@tf.function
def hypo(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

@tf.function
def cost(X, Y):
    hy_val = hypo(X)
    return tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hy_val), axis=1))

@tf.function
def predict(X):
    hy_val = hypo(X)
    return tf.math.argmax(hy_val, 1)

@tf.function
def accuracy(predict, Y):
    is_correct = tf.equal(predict, tf.math.argmax(Y, 1))

    return tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

X = tf.Variable(x_data, dtype=tf.float32, shape=[None, 3])
Y = tf.Variable(y_data, dtype=tf.float32, shape=[None, 3])

X2 = tf.Variable(x_test, dtype=tf.float32, shape=[None, 3])
Y2 = tf.Variable(y_test, dtype=tf.float32, shape=[None, 3])

# 모델의 성능 평가
for step in range(20):
    with tf.GradientTape() as tp:
        cost_val = cost(X, Y)

    train = optimizer.minimize(cost_val, var_list=[W, b], tape=tp)
    print(f'STEP = {step:>03}, cost_val = {cost_val}, \n\tW_val = {W.numpy()}')

# 테스트 데이터로 결과 예측

pd = predict(X2)
print(f'Prediction: {pd}')
print(f'Accuracy: {accuracy(pd, Y2)}')
```

- 결과

```cmd
D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercise_lesson6_1.py
STEP = 000, cost_val = 5.086367607116699, 
	W_val = [[ 0.5995157  -1.1033959   1.045647  ]
 [ 0.44489798  1.1227759   0.7363239 ]
 [ 1.5081984   0.8920087   1.2719026 ]]
STEP = 001, cost_val = 2.031190872192383, 
	W_val = [[ 0.584086  -1.0680851  1.0257679]
 [ 0.4429418  1.2870353  0.5740101]
 [ 1.4928694  1.0703707  1.1088624]]
STEP = 002, cost_val = 1.3068029880523682, 
	W_val = [[ 0.5526987 -1.0511924  1.0402745]
 [ 0.3730932  1.3396244  0.5913105]
 [ 1.4043013  1.1497247  1.1181153]]
STEP = 003, cost_val = 1.2447235584259033, 
	W_val = [[ 0.5453317 -1.0524024  1.0488477]
 [ 0.4258642  1.2862232  0.5919216]
 [ 1.4376539  1.1285837  1.1058755]]
STEP = 004, cost_val = 1.208823323249817, 
	W_val = [[ 0.52739346 -1.0442824   1.0586591 ]
 [ 0.4245916   1.2866199   0.59276086]
 [ 1.4169477   1.1584123   1.0967202 ]]
STEP = 005, cost_val = 1.192062258720398, 
	W_val = [[ 0.51372313 -1.0401686   1.0682172 ]
 [ 0.44489646  1.2640178   0.59505576]
 [ 1.4179136   1.1660821   1.0880868 ]]
STEP = 006, cost_val = 1.1772292852401733, 
	W_val = [[ 0.498031  -1.0341903  1.0779325]
 [ 0.4538564  1.2525623  0.5975323]
 [ 1.4079343  1.1839708  1.080163 ]]
STEP = 007, cost_val = 1.1633715629577637, 
	W_val = [[ 0.48353666 -1.0292274   1.0874647 ]
 [ 0.46816456  1.2356014   0.60019034]
 [ 1.4036115   1.1962343   1.0722163 ]]
STEP = 008, cost_val = 1.1495609283447266, 
	W_val = [[ 0.46864104 -1.023767    1.0969038 ]
 [ 0.479392    1.2219044   0.6026577 ]
 [ 1.3966535   1.2111849   1.0642344 ]]
STEP = 009, cost_val = 1.1364059448242188, 
	W_val = [[ 0.4541223 -1.0185828  1.1062315]
 [ 0.4916799  1.2070392  0.6052135]
 [ 1.3911009  1.2246011  1.0563349]]
STEP = 010, cost_val = 1.1234368085861206, 
	W_val = [[ 0.4396997  -1.0133849   1.1154608 ]
 [ 0.5035268   1.1926578   0.60777473]
 [ 1.3855009   1.238054    1.0485048 ]]
STEP = 011, cost_val = 1.110884189605713, 
	W_val = [[ 0.4253244 -1.0081466  1.1246008]
 [ 0.5146886  1.1789494  0.6103382]
 [ 1.3795912  1.2516999  1.0407778]]
STEP = 012, cost_val = 1.098517656326294, 
	W_val = [[ 0.4111795  -1.0030471   1.1336485 ]
 [ 0.5261266   1.1648853   0.61298335]
 [ 1.3743391   1.2645738   1.0331904 ]]
STEP = 013, cost_val = 1.086612343788147, 
	W_val = [[ 0.39713228 -0.99797815  1.1426222 ]
 [ 0.5371374   1.1511371   0.6157116 ]
 [ 1.3690656   1.2772752   1.025741  ]]
STEP = 014, cost_val = 1.0750725269317627, 
	W_val = [[ 0.38319948 -0.99292064  1.1514945 ]
 [ 0.5478109   1.1377857   0.6183781 ]
 [ 1.3638303   1.2899026   1.0183306 ]]
STEP = 015, cost_val = 1.0638052225112915, 
	W_val = [[ 0.36940706 -0.9878608   1.1602355 ]
 [ 0.5582495   1.1249393   0.6208569 ]
 [ 1.3587842   1.3025323   1.0108179 ]]
STEP = 016, cost_val = 1.0527567863464355, 
	W_val = [[ 0.35592133 -0.9830009   1.1688675 ]
 [ 0.5693182   1.1114506   0.6232815 ]
 [ 1.3547711   1.3140572   1.0033274 ]]
STEP = 017, cost_val = 1.042025089263916, 
	W_val = [[ 0.3422922  -0.97801125  1.1775123 ]
 [ 0.5787542   1.099239    0.62606394]
 [ 1.3495282   1.3263116   0.996332  ]]
STEP = 018, cost_val = 1.0318320989608765, 
	W_val = [[ 0.3289972  -0.97319174  1.1859893 ]
 [ 0.58898216  1.086578    0.62852746]
 [ 1.3454586   1.3376397   0.98909706]]
STEP = 019, cost_val = 1.0218462944030762, 
	W_val = [[ 0.31574342 -0.9683608   1.1944121 ]
 [ 0.59849143  1.0745114   0.631062  ]
 [ 1.3410977   1.349038    0.9820521 ]]
Prediction: [2 2 2]
Accuracy: 1.0
```

### Learning rate가 작은 경우

- learning rate를 15로 만들면 cost function 값이 발산 -> nan

  - `optimizer = tf.keras.optimizers.SGD(learning_rate=15)`

- STEP=001부터 cost 값이 발산

  - ```cmd
    STEP = 000, cost_val = 7.121092796325684, 
    	W_val = [[-12.051267    6.373541    4.8394084]
     [-40.144817   26.467255   14.396875 ]
     [-41.078876   25.790483   12.242223 ]]
    STEP = 001, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 002, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 003, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 004, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 005, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 006, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 007, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 008, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 009, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 010, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 011, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 012, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 013, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 014, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 015, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 016, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 017, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 018, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    STEP = 019, cost_val = nan, 
    	W_val = [[nan nan nan]
     [nan nan nan]
     [nan nan nan]]
    Prediction: [0 0 0]
    Accuracy: 0.0
    ```

### Learning rate가 작은 경우

- learning rate를 1e-15로 만들면 cost 함수값이 변화 없음 -> 학습 진행 안 됨

  - `optimizer = tf.keras.optimizers.SGD(learning_rate=1e-15)`

- 두번째 학습부터 cost 함수값이 고정(수렴)

  - ```cmd
    D:\engine\venv\Scripts\python.exe D:/Python_Projects/SampleProject1/exercise_lesson6_1.py
    STEP = 000, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 001, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 002, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 003, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 004, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 005, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 006, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 007, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 008, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 009, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 010, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 011, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 012, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 013, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 014, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 015, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 016, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 017, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 018, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    STEP = 019, cost_val = 8.578227996826172, 
    	W_val = [[-0.18087557  1.377337   -0.28001866]
     [ 0.12942733  1.2387561   1.5038089 ]
     [-0.45850936 -0.57692444  1.2089869 ]]
    Prediction: [1 1 2]
    
    Accuracy: 0.3333333432674408
    ```

