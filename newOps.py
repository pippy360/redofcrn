def UnPooling2x2ZeroFilled(x):
  # https://github.com/tensorflow/tensorflow/issues/2169
  out = tf.concat([x, tf.zeros_like(x)], 3)
  out = tf.concat([out, tf.zeros_like(out)], 2)

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
      out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
      return tf.reshape(out, out_size)
  else:
      shv = tf.shape(x)
      ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
  return ret

def main():
  inputs = inputs
  conv = inputs
  conv = UnPooling2x2ZeroFilled(conv)
  conv = 5x5conv
  #do this 4 times.....???? copy the functions and still implement it...
  
  
  
