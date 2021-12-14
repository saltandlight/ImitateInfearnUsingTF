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
    print(str.numpy())
    ```

  - ```cmd
    b'Hello, TensorFlow!'
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
    print(node4.numpy())
    ```

  - 결과:

    - ```cmd
      7.0
      ```

- TensorFlow에서 변수 값 바꾸기

  - ```python
    import tensorflow as tf
    
    @tf.function
    def add(a, b):
        return tf.add(a, b)
    
    a = tf.Variable(3, dtype=tf.float32, name="a")
    b = tf.Variable(4.5, dtype=tf.float32, name="b")
    
    adder_node = add(a, b)
    print(adder_node.numpy())
    
    adder_node2 = add([1, 3], [2, 4])
    print(adder_node2.numpy())
    ```

## Tensor 란 무엇인가

- 텐서: array, 어떤 값이든 될 수 있음
- Rank, Shape, Type을 사용
  - Rank: 몇 차원 배열인가?
  - Shape: 각각의 원소에 몇 개가 들어있는가?
    - 텐서 설계 시 매우 중요 => 몇 개의 원소 가지고 있는지 잘 살펴봐야 함
    - 텐서 shape는 리스트일 수도 있고 튜플일 수도 있음
    - 차원의 개수만큼 숫자들이 리스트 안에 있을 수 있음, 리스트 안의 각 숫자 = 각 차원의 숫자
      - 각 차원의 값 = None, 길이 가변
      - tf.shape() 연산으로 모양 확인 가능
  - Type: 데이터의 종류
    - 대부분의 경우 float32를 사용
  - 연산(OP): 텐서 객체에 계산 수행하는 점(node)
    - 0개 이상의 input -> 0개 이상의 텐서 return
    - input, output이 모두 0개인 연산
      - tensorflow에서 연산은 단순한 수학적인 계산 이상, 상태 초기화 등에도 사용됨
      - 모든 점들이 다른 점들과 반드시 연결될 필요 없음
    - 연산의 생성자는 문자열 매개변수인 name을 입력으로 받아들임
      - name 매개변수를 사용하여 문자열로 특정 연산 참조 가능
      - `c = tf.add(a, b, name = 'add_c')`

참고: `https://forensics.tistory.com/5?category=767888`

