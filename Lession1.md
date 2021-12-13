# TensorFlow 개념잡기 연습 따라하기(tf1 to tf2)

## Tensor

- 텐서: 2차원 행렬(2x2)

## Data Flow Graph

- 데이터 흐름 그래프: 데이터들이 점 지나면서 연산 이루어짐, 내가 원하는 결과 얻거나 작업이 이루어지는 과정으로 텐서가 그래프 상에서 돌아다님 (= 텐서가 흐름 =  Tensor Flow)

## TensorFlow 실행

- TensorFlow 설치

- ```python
  >>> import tensorflow as tf
  >>> tf.__version__
  '2.7.0'
  ```

- Hello, TensorFlow! 출력하기

  - tensorflow v2에서는 `Session` 을 만들지 않고 `tf.function`을 사용합니다.

  - 자세한 설명: `https://www.tensorflow.org/guide/migrate?hl=ko#1_tfsessionrun_%ED%98%B8%EC%B6%9C%EC%9D%84_%EB%B0%94%EA%BE%B8%EC%84%B8%EC%9A%94`

  - ```python
    import tensorflow as tf
    
    @tf.function
    def hello():
        return tf.constant("Hello, TensorFlow!")
    
    str = tf.function(hello)()
    print(str)
    ```

  - ```cmd
    tf.Tensor(b'Hello, TensorFlow!', shape=(), dtype=string)
    ```

- 2개의 데이터 받아서 더하는 그래프 만들기

  - leaf node의 텐서들이 각각 선을 따라 이동해서 연산을 통하여 결과값 텐서를 만들어냄

  - ```python
    >>> node1 = tf.constant(3.0, tf.float32)
    >>> node2 = tf.constant(4.0)
    >>> node3 = tf.add(node1, node2)
    >>> print("점1 =  ", node1, "점2 = ", node2)
    점1 =   tf.Tensor(3.0, shape=(), dtype=float32) 점2 =  tf.Tensor(4.0, shape=(), dtype=float32)
    >>> print("점3=", node3)
    점3= tf.Tensor(7.0, shape=(), dtype=float32)
    ```

  - ```python
    import tensorflow as tf
    
    @tf.function
    def add(a, b):
        return a + b
    
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node4 = add(node1, node2)
    print(node4)
    ```

  - 결과:

    - ```cmd
      tf.Tensor(7.0, shape=(), dtype=float32)
      ```

참고: `https://forensics.tistory.com/5?category=767888`

