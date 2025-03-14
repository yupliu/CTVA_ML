import torch

x = torch.tensor(32)

print(type(x))

print(x.shape)


import tensorflow as tf

x = tf.Variable(32,tf.int32)

y = tf.Variable(25,tf.int32)

print(x+y)