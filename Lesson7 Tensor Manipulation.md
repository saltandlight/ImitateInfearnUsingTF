# Lesson 7: Tensor Manipulation

## Shape, Rank, Axis

### Tensor의 Shape와 Rank

```python
>>> import tensorflow as tf
>>> t1 = tf.constant([1, 2, 3, 4])
2021-12-17 14:32:13.364034: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-17 14:32:14.169537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
>>> tf.shape(t1)
<tf.Tensor: shape=(1,), dtype=int32, numpy=array([4])>
>>> tf.rank(t1)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
>>> tf.rank(t1).numpy()
1
>>> t2 = tf.constant([[1, 2],
...                   [3, 4]])
>>> tf.shape(t2)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 2])>
>>> tf.shape(t2).numpy()
array([2, 2])
>>> tf.rank(t2).numpy()
2
>>> t3 = tf.constant([
...                   [
...                    [
...                     [1, 2, 3, 4],
...                     [5, 6, 7, 8],
...                     [9, 10, 11, 12]
...                     ],
...                     [
...                      [13, 14, 15, 16],
...                      [17, 18, 19, 20],
...                      [21, 22, 23, 24]
...                     ]
...                    ]
...                   ])
>>> tf.shape(t3).numpy()
array([1, 2, 3, 4])
>>> tf.rank(t3).numpy()
4
```

### Tensor에서 matmul() 함수와 multiply() 함수의 차이

- matmul() 함수

```python
>>> import tensorflow as tf
>>> matrix1 = tf.constant([
...                        [1, 2],
...                        [3, 4]
...                       ])
2021-12-17 14:52:21.937255: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-17 14:52:22.285872: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5468 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6
>>> matrix2 = tf.constant([
...                        [1],
...                        [2],
...                       ])
>>> matrix3 = tf.constant([[1, 2]])
>>> print(f'Matrix 1\'s shape is {matrix1.shape}')
Matrix 1's shape is (2, 2)
>>> print(f'Matrix 2\'s shape is {matrix2.shape}')
Matrix 2's shape is (2, 1)
>>> print(f'Matrix 3\'s shape is {matrix3.shape}')
Matrix 3's shape is (1, 2)
>>> print(tf.matmul(matrix1, matrix2))
tf.Tensor(
[[ 5]
 [11]], shape=(2, 1), dtype=int32)
>>> print(tf.matmul(matrix3, matrix1).numpy())
[[ 7 10]]
>>> print(tf.matmul(matrix1, matrix2).numpy())
[[ 5]
 [11]]
>>> print(tf.matmul(matrix1, matrix3).numpy())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\user\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow\python\util\traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "C:\Users\user\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow\python\framework\ops.py", line 7107, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
tensorflow.python.framework.errors_impl.InvalidArgumentError: Matrix size-incompatible: In[0]: [2,2], In[1]: [1,2] [Op:MatMul]
```

- multiply() 함수

```python
>>> print(tf.multiply(matrix1, matrix2))
tf.Tensor(
[[1 2]
 [6 8]], shape=(2, 2), dtype=int32)
>>> print(tf.multiply(matrix3, matrix1).numpy())
[[1 4]
 [3 8]]
>>> print(tf.multiply(matrix1, matrix2).numpy())
[[1 2]
 [6 8]]
>>> print(tf.multiply(matrix1, matrix3).numpy())
[[1 4]
 [3 8]]
```

- reduce_mean() 함수

```python
>>> t1 = tf.constant([1., 2.])
>>> t2 = tf.constant([[1., 2.],
...                   [3., 4.]])
>>> print(tf.reduce_mean(t1).numpy())
1.5
>>> print(tf.reduce_mean(t2, axis=0).numpy())
[2. 3.]
>>> print(tf.reduce_mean(t2, axis=1).numpy())
[1.5 3.5]
>>> print(tf.reduce_mean(t2, axis=-1).numpy())
[1.5 3.5]
```

- reduce_sum() 함수

```python
>>> t = tf.constant([[1., 2.],
...                  [3., 4.]])
>>> print(tf.reduce_sum(t, axis=0).numpy()
... )
[4. 6.]
>>> print(tf.reduce_sum(t, axis=1).numpy())
[3. 7.]
>>> print(tf.reduce_mean(tf.reduce_sum(t, axis=0)).numpy())
5.0
>>> print(tf.reduce_mean(tf.reduce_sum(t, axis=1)).numpy())
5.0
```

- argmax() 함수와 argmin() 함수

```python
>>> t = tf.constant([[1., 4., 5., 6.],
...                  [3., 2., 6., 6.]])
>>> print(tf.argmax(t, axis=0).numpy())
[1 0 1 0]
>>> print(tf.argmax(t, axis=1).numpy())
[3 2]
>>> print(tf.argmin(t, axis=0).numpy())
[0 1 0 0]
>>> print(tf.argmin(t, axis=1).numpy())
[0 1]
```

- squeeze() 함수

```python
>> t = tf.constant([[1], [2], [3]])
>>> print(tf.rank(t).numpy())
2
>>> s = tf.squeeze(t)
>>> print(s.numpy())
[1 2 3]
>>> print(tf.rank(s).numpy())
1
```

- expaned_dims() 함수: tensor의 dimension 하나 늘림

```python
>>> t = tf.constant([[1], [2], [3]])
>>> print(tf.rank(t).numpy())
2
>>> s = tf.squeeze(t)
>>> print(s.numpy())
[1 2 3]
>>> print(tf.rank(s).numpy())
1
>>> t = tf.constant([1, 2, 3, 4])
>>> print(tf.rank(t).numpy())
1
>>> s = tf.expand_dims(t, axis=0)
>>> print(s.numpy())
[[1 2 3 4]]
>>> print(tf.rank(s).numpy())
2
>>> s = tf.expand_dims(t, axis=1)
>>> print(s.numpy())
[[1]
 [2]
 [3]
 [4]]
>>> print(tf.rank(s).numpy())
2
```

- one_hot() 함수: tensor의 원소값에 해당하는 index의 값만 1이고 나머지는 0인 tensor로 반환
  - 매개변수 depth는 만들고자 하는 tensor의 shape 중 가장 마지막 요소의 값

```python
>>> t = tf.constant([[1], [0], [2], [1], [3]])
>>> s = tf.one_hot(t, depth=4)
>>> print(s.numpy())
[[[0. 1. 0. 0.]]

 [[1. 0. 0. 0.]]

 [[0. 0. 1. 0.]]

 [[0. 1. 0. 0.]]

 [[0. 0. 0. 1.]]]
>>> print(tf.shape(s).numpy())
[5 1 4]
>>> s = tf.one_hot(t, depth=6)
>>> print(s.numpy())
[[[0. 1. 0. 0. 0. 0.]]

 [[1. 0. 0. 0. 0. 0.]]

 [[0. 0. 1. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0.]]

 [[0. 0. 0. 1. 0. 0.]]]
>>> print(tf.shape(s).numpy())
[5 1 6]
```

