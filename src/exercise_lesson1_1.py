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