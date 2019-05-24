import numpy as np
import tensorflow as tf
import time
import random


print(tf.executing_eagerly())
tf.debugging.set_log_device_placement(True)


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{} function took {} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap


@timing
def python_massive_calculations(vector_size=100000):
    Vector_1 = [random.random() for x in range(vector_size)]
    Vector_2 = [random.random() for x in range(vector_size)]
    return list(map(lambda x, y: x * y, Vector_1, Vector_2))


@timing
def numpy_massive_calculations(vector_size=100000):
    Vector_1 = np.random.random(vector_size)
    Vector_2 = np.random.random(vector_size)
    return np.multiply(Vector_1, Vector_2)


@timing
def tf_massive_calculations(vector_size=100000):
    Vector_1 = tf.constant(np.random.random(vector_size))
    Vector_2 = tf.constant(np.random.random(vector_size))
    return tf.math.multiply(Vector_1, Vector_2)


def test():
    vectors_length = 10000000
    pyvec = python_massive_calculations(vectors_length)
    print(pyvec[0:10])
    print(type(pyvec[0]))
    print(len(pyvec))
    nuvec = numpy_massive_calculations(vectors_length)
    print(nuvec[0:10])
    print(type(nuvec[0]))
    print(len(nuvec))
    tfvec = tf_massive_calculations(vectors_length)
    print(tfvec[0:10])
    print(type(tfvec[0]))
    print(len(tfvec))


if __name__ == "__main__":
    test()
