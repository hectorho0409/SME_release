# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):


  def _repeat(x, n_repeats):
    
    rep = tf.transpose(tf.expand_dims(tf.ones([n_repeats]), 1), [1, 0])
    rep = tf.cast(rep, 'int32')

    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


  def _interpolate(im, x, y, out_size):

    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    tf.cast(height, 'float32')
    tf.cast(width, 'float32')

    out_height = out_size[0]
    out_width = out_size[1]

    zero = int(0)

    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x1_fake = x1
    y1_fake = y1
    
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width
    dim1 = width*height
    base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, (-1, channels))
    im_flat = tf.cast(im_flat, 'float32')

    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')

    x1_fake = tf.cast(x1_fake, 'float32')
    y1_fake = tf.cast(y1_fake, 'float32')
    
    da = tf.square((x1_fake - x) * (y1_fake - y))
    db = tf.square((x1_fake - x) * (y - y0_f))
    dc = tf.square((x - x0_f) * (y1_fake - y))
    dd = tf.square((x - x0_f) * (y - y0_f))
    weight_base = (da + db + dc + dd)
    wa = tf.expand_dims(da / weight_base, 1)
    wb = tf.expand_dims(db / weight_base, 1)
    wc = tf.expand_dims(dc / weight_base, 1)
    wd = tf.expand_dims(dd / weight_base, 1)

    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return output


  def _meshgrid(height, width):
    
    x_t = tf.matmul(tf.ones([height, 1]),tf.transpose(tf.expand_dims(tf.linspace(0.0, width-1, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, height-1, height), 1),tf.ones([1, width]))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    ones = tf.ones_like(x_t_flat)
    grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
    
    return grid

  
  def _transform(theta, input_dim, out_size):

    num_batch = tf.shape(input_dim)[0]
    height = tf.shape(input_dim)[1]
    width = tf.shape(input_dim)[2]
    num_channels = tf.shape(input_dim)[3]

    theta = tf.reshape(theta, (-1, 2, 3))
    theta = tf.cast(theta, 'float32')
    
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')

    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width)

    grid = tf.expand_dims(grid, 0)
    grid = tf.reshape(grid, [-1])
    
    grid = tf.tile(grid, tf.stack([num_batch]))
    grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = tf.matmul(theta, grid)

    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

    x_s_flat = tf.reshape(x_s, [-1])
    y_s_flat = tf.reshape(y_s, [-1])

    input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat,out_size)

    output = tf.reshape(input_transformed, (num_batch, out_height*out_width))
    
    return output

  
  output = _transform(theta, U, out_size)
  return output
