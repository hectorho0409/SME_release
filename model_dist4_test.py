import numpy as np
import tensorflow as tf
import os
from utils import bilinear_interp, meshgrid

class PolicyEstimator(object):
    """
    Policy Function approximator. 
    """
    def __init__(self, height, width, downsample):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            self.sess = tf.InteractiveSession()
            block_num = int(height*width/(downsample**2))
            down_height, down_width = int(height/downsample), int(width/downsample)
            
            # ====================================
            # hardware acquirements
            # ====================================
            self.training_process_num = 1  # group number
            
            # ====================================
            # play part (random or max)
            # ====================================
            self.batch_size_play = tf.placeholder(tf.int32)
            
            # input part
            self.state = tf.placeholder(tf.float32, shape=(None, height, width, 4))
            self.state_pad = tf.placeholder(tf.float32, shape=(None, height+64, width+64, 15))
            self.mask = tf.placeholder(tf.float32, shape=(None, down_height, down_width, 1))
            self.mot_state = tf.placeholder(tf.float32, shape=(None, 2))
            self.context_frame = tf.placeholder(tf.float32, shape=(None, 4, height, width, 3))
            self.target_frame = tf.placeholder(tf.float32, shape=(None, 1, height, width, 3))
            self.region_adaptive_pos = tf.placeholder(tf.float32, shape=(None, 2))
            self.region_adaptive_mot = tf.placeholder(tf.float32, shape=(None, 2))
            
            # model - position part
            self.play_pos_feature, oldpi_parms_position = self.position_model(self.state, self.mask, "pi", trainable=True, reuse_model=None)
            # model - region adaptive part
            self.region_adaptive_feature, oldpi_parms_region_adaptive = self.region_adaptive_model(self.region_adaptive_pos, self.region_adaptive_mot, self.state_pad, self.batch_size_play, "pi", trainable=True, reuse_model=None, reuse_core=None)
            # model - motion part
            self.play_mot, self.diff, self.diff_f1, self.diff_f2, self.diff_f3, oldpi_parms_motion = self.motion_model(self.state_pad, self.mot_state, self.batch_size_play, "pi", trainable=True, reuse_model=None, reuse_core=True)
            
            # get position from model
            self.play_pos_probs = tf.nn.softmax(self.play_pos_feature)
            self.pos_choice_random = tf.multinomial(1.*self.play_pos_feature, 1)
            self.pos_height_random = tf.div(self.pos_choice_random, down_width)
            self.pos_width_random  = tf.floormod(self.pos_choice_random, down_width)
            self.pos_choice_max = tf.expand_dims(tf.argmax(self.play_pos_probs, axis = -1), 1)
            self.pos_height_max = tf.div(self.pos_choice_max, down_width)
            self.pos_width_max  = tf.floormod(self.pos_choice_max, down_width)

            # get region_adaptive from model
            self.play_region_adaptive_probs = tf.nn.softmax(self.region_adaptive_feature)
            self.region_adaptive_choice_random = tf.multinomial(1.*self.region_adaptive_feature, 1)
            self.region_adaptive_choice_random_onehot = tf.one_hot(indices = self.region_adaptive_choice_random, depth = 7, on_value = 1., off_value = 0.)
            self.order_choice_random = tf.reduce_sum(tf.multiply(self.region_adaptive_choice_random_onehot, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]), -1)
            self.region_adaptive_choice_max = tf.expand_dims(tf.argmax(self.region_adaptive_feature, axis=-1), 1)
            self.region_adaptive_choice_max_onehot = tf.one_hot(indices = self.region_adaptive_choice_max, depth = 7, on_value = 1., off_value = 0.)
            self.order_choice_max = tf.reduce_sum(tf.multiply(self.region_adaptive_choice_max_onehot, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]), -1)
            
            # greedy position
            self.greedy_height, self.greedy_width = self.greedy_pos(height, width)
            
            # ====================================
            # others..
            # ====================================
            #initialize all tensor variable parameters
            self.sess.run(tf.global_variables_initializer())
            # build saver
            self.saver_position = tf.train.Saver(oldpi_parms_position)
            self.saver_motion = tf.train.Saver(oldpi_parms_motion)
            self.saver_region_adaptive = tf.train.Saver(oldpi_parms_region_adaptive)
    
    
    def position_model(self, state, mask, scope, trainable, reuse_model):
        with tf.variable_scope(scope):
            
            # X1: residual, X2: image
            X1 = tf.slice(state, [0, 0, 0, 0], [-1, -1, -1, 3])
            X2 = tf.slice(state, [0, 0, 0, 3], [-1, -1, -1, 1])

            # -----------------------------------------
            # position model part
            # -----------------------------------------
            # residual encode part (position-img)
            enc_1 = tf.layers.conv2d(inputs=    X1, filters=  16, kernel_size=(7, 7), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos1_1")
            enc_1 = tf.contrib.layers.max_pool2d(enc_1, [2, 2])

            enc_2 = tf.layers.conv2d(inputs= enc_1, filters=  32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos1_2")
            enc_2 = tf.contrib.layers.max_pool2d(enc_2, [2, 2])

            enc_3 = tf.layers.conv2d(inputs= enc_2, filters= 64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos1_3")
            enc_3 = tf.contrib.layers.max_pool2d(enc_3, [2, 2])

            enc_4 = tf.layers.conv2d(inputs=enc_3, filters= 128, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos1_4")
            enc_4 = tf.contrib.layers.max_pool2d(enc_4, [2, 2])

            # residual encode part (position-flow)
            fenc_1 = tf.layers.conv2d(inputs=     X2, filters=  16, kernel_size=(7, 7), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos2_1")
            fenc_1 = tf.contrib.layers.max_pool2d(fenc_1, [2, 2])   #144

            fenc_2 = tf.layers.conv2d(inputs= fenc_1, filters=  32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos2_2")
            fenc_2 = tf.contrib.layers.max_pool2d(fenc_2, [2, 2])   #72

            fenc_3 = tf.layers.conv2d(inputs= fenc_2, filters= 64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos2_3")
            fenc_3 = tf.contrib.layers.max_pool2d(fenc_3, [2, 2])   #36

            fenc_4 = tf.layers.conv2d(inputs= fenc_3, filters= 128, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos2_4")
            fenc_4 = tf.contrib.layers.max_pool2d(fenc_4, [2, 2])   #18

            pos_encode = tf.concat(values = [enc_4, fenc_4], axis = -1)

            # residual decode part (position)
            dec_1 = tf.image.resize_bilinear(pos_encode, [36, 44])
            #dec_1 = tf.image.resize_bilinear(pos_encode, [30, 40])
            dec_1 = tf.layers.conv2d_transpose(inputs= dec_1, filters= 64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos3_1")
            dec_2 = tf.image.resize_bilinear(dec_1, [72, 88])
            #dec_2 = tf.image.resize_bilinear(dec_1, [60, 80])
            dec_2 = tf.layers.conv2d_transpose(inputs= dec_2, filters=  1, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="pos3_2")

            inverse_mask = tf.multiply(dec_2, 1. - mask)
            inverse_mask -= mask
            
            # mask version feature (for play and predict)
            shape = inverse_mask.get_shape().as_list()
            pos_feature_mask  = tf.reshape(inverse_mask, [-1, reduce(lambda x, y: x * y, shape[1:])])
            
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        return 50.*pos_feature_mask, params
    
    
    def interpolation(self, input, trainable, reuse_core):

        interpolation_1 = tf.layers.conv2d(inputs=input, filters=8, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_1")
        interpolation_1 = tf.contrib.layers.max_pool2d(interpolation_1, [2, 2])  # 32

        interpolation_2 = tf.layers.conv2d(inputs=interpolation_1, filters=16, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_2")

        interpolation_3 = tf.layers.conv2d(inputs=interpolation_2, filters=32, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_3")
        interpolation_3 = tf.contrib.layers.max_pool2d(interpolation_3, [2, 2])  # 16

        interpolation_4 = tf.layers.conv2d(inputs=interpolation_3, filters=64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_4")

        interpolation_5 = tf.layers.conv2d(inputs=interpolation_4, filters=128, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_5")
        interpolation_5 = tf.contrib.layers.max_pool2d(interpolation_5, [2, 2])  # 8

        interpolation_6 = tf.layers.conv2d(inputs=interpolation_5, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_6")
        interpolation_7 = tf.layers.conv2d(inputs=interpolation_6, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_7")
        interpolation_8 = tf.layers.conv2d(inputs=interpolation_7, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_8")
        interpolation_9 = tf.layers.conv2d(inputs=interpolation_8, filters=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_9")
        interpolation_10 = tf.layers.conv2d(inputs=interpolation_9, filters=8, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_10")
        interpolation_11 = tf.layers.conv2d(inputs=interpolation_10, filters=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_core, name="interpolation_11")
        flow = tf.reduce_mean(interpolation_11, [1, 2])

        return flow
    
    
    def motion_model(self, state_pad, mot_state, batch_size, scope, trainable, reuse_model, reuse_core):
        with tf.variable_scope(scope):
            
            # X1: residual, X2: image
            #X1 = tf.slice(state, [0, 0, 0, 0], [-1, -1, -1, 3])
            #X2 = tf.slice(state, [0, 0, 0, 3], [-1, -1, -1, 1])
            X3 = tf.slice(state_pad, [0, 0, 0, 0], [-1, -1, -1, 12])
            X4 = tf.slice(state_pad, [0, 0, 0, 0], [-1, -1, -1, 6])
            X5 = tf.slice(state_pad, [0, 0, 0, 3], [-1, -1, -1, 6])
            X6 = tf.slice(state_pad, [0, 0, 0, 6], [-1, -1, -1, 6])

            '''
            img_tar = tf.reshape(target_frame,[batch_size,240,320,3])
            img_last = tf.reshape(context_frame[:,0,:,:,:],[batch_size,240,320,3])

            img_tar_f1 = tf.reshape(context_frame[:,2,:,:,:],[batch_size,240,320,3])
            img_last_f1 = tf.reshape(context_frame[:,3,:,:,:],[batch_size,240,320,3])

            img_tar_f2 = tf.reshape(context_frame[:,1,:,:,:],[batch_size,240,320,3])
            img_last_f2 = tf.reshape(context_frame[:,2,:,:,:],[batch_size,240,320,3])

            img_tar_f3 = tf.reshape(context_frame[:,0,:,:,:],[batch_size,240,320,3])
            img_last_f3 = tf.reshape(context_frame[:,1,:,:,:],[batch_size,240,320,3])
            '''

            img_tar = tf.slice(state_pad, [0, 0, 0, 12], [-1, -1, -1, 3])
            img_last = tf.slice(state_pad, [0, 0, 0, 9], [-1, -1, -1, 3])

            img_tar_f1 = tf.slice(state_pad, [0, 0, 0, 3], [-1, -1, -1, 3])
            img_last_f1 = tf.slice(state_pad, [0, 0, 0, 0], [-1, -1, -1, 3])

            img_tar_f2 = tf.slice(state_pad, [0, 0, 0, 6], [-1, -1, -1, 3])
            img_last_f2 = tf.slice(state_pad, [0, 0, 0, 3], [-1, -1, -1, 3])

            img_tar_f3 = tf.slice(state_pad, [0, 0, 0, 9], [-1, -1, -1, 3])
            img_last_f3 = tf.slice(state_pad, [0, 0, 0, 6], [-1, -1, -1, 3])

            # -----------------------------------------
            # motion model part
            # -----------------------------------------
            x = tf.slice(mot_state, [0, 0], [-1, 1]) * 4.
            y = tf.slice(mot_state, [0, 1], [-1, 1]) * 4.
            
            """
            x1 = (x - 1. * 32. + 32.) / (240. + 64.)
            x2 = (x + 1. * 32. + 32.) / (240. + 64.)
            y1 = (y - 1. * 32. + 32.) / (320. + 64.)
            y2 = (y + 1. * 32. + 32.) / (320. + 64.)
            """
            x1 = (x - 1. * 32. + 32.) / (288. + 64.)
            x2 = (x + 1. * 32. + 32.) / (288. + 64.)
            y1 = (y - 1. * 32. + 32.) / (352. + 64.)
            y2 = (y + 1. * 32. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)

            mot_state_1 = tf.image.crop_and_resize(X3, boxes, tf.range(batch_size), [64, 64])

            mot_flow_1 = tf.image.crop_and_resize(X4, boxes, tf.range(batch_size), [64, 64])
            mot_flow_2 = tf.image.crop_and_resize(X5, boxes, tf.range(batch_size), [64, 64])
            mot_flow_3 = tf.image.crop_and_resize(X6, boxes, tf.range(batch_size), [64, 64])

            ####F1
            f1 = self.interpolation(mot_flow_1, trainable=True, reuse_core=reuse_core)
            flow1 = tf.reshape(f1, [batch_size, 1, 1, 2])
            flow1_state = tf.tile(flow1, [1, 8, 8, 1])
            #flow1 = tf.tile(flow1, [1, 240 + 64, 320 + 64, 1])
            flow1 = tf.tile(flow1, [1, 288 + 64, 352 + 64, 1])

            #grid_x, grid_y = meshgrid(240 + 64, 320 + 64)
            grid_x, grid_y = meshgrid(288 + 64, 352 + 64)

            grid_x = tf.tile(grid_x, [batch_size, 1, 1])  # batch_size = 100
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])  # batch_size = 100

            coor_x = grid_x + flow1[:, :, :, 0]
            coor_y = grid_y + flow1[:, :, :, 1]

            shifted = bilinear_interp(img_last_f1, coor_x, coor_y, 'interpolate')
            
            """
            x1 = (x - 1. * 6. + 32.) / (240. + 64.)
            x2 = (x + 1. * 6. + 32.) / (240. + 64.)
            y1 = (y - 1. * 6. + 32.) / (320. + 64.)
            y2 = (y + 1. * 6. + 32.) / (320. + 64.)
            """
            x1 = (x - 1. * 6. + 32.) / (288. + 64.)
            x2 = (x + 1. * 6. + 32.) / (288. + 64.)
            y1 = (y - 1. * 6. + 32.) / (352. + 64.)
            y2 = (y + 1. * 6. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)

            shifted_crop = tf.image.crop_and_resize(shifted, boxes, tf.range(batch_size), [12, 12])
            img_tar_crop = tf.image.crop_and_resize(img_tar_f1, boxes, tf.range(batch_size), [12, 12])
            crop_diff_flow1 = shifted_crop - img_tar_crop

            ####F2
            f2 = self.interpolation(mot_flow_2, trainable=True, reuse_core=reuse_core)
            flow2 = tf.reshape(f2, [batch_size, 1, 1, 2])
            flow2_state = tf.tile(flow2, [1, 8, 8, 1])
            #flow2 = tf.tile(flow2, [1, 240 + 64, 320 + 64, 1])
            flow2 = tf.tile(flow2, [1, 288 + 64, 352 + 64, 1])
            
            #grid_x, grid_y = meshgrid(240 + 64, 320 + 64)
            grid_x, grid_y = meshgrid(288 + 64, 352 + 64)

            grid_x = tf.tile(grid_x, [batch_size, 1, 1])  # batch_size = 100
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])  # batch_size = 100

            coor_x = grid_x + flow2[:, :, :, 0]
            coor_y = grid_y + flow2[:, :, :, 1]

            shifted = bilinear_interp(img_last_f2, coor_x, coor_y, 'interpolate')
            
            """
            x1 = (x - 1. * 6. + 32.) / (240. + 64.)
            x2 = (x + 1. * 6. + 32.) / (240. + 64.)
            y1 = (y - 1. * 6. + 32.) / (320. + 64.)
            y2 = (y + 1. * 6. + 32.) / (320. + 64.)
            """
            x1 = (x - 1. * 6. + 32.) / (288. + 64.)
            x2 = (x + 1. * 6. + 32.) / (288. + 64.)
            y1 = (y - 1. * 6. + 32.) / (352. + 64.)
            y2 = (y + 1. * 6. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)

            shifted_crop = tf.image.crop_and_resize(shifted, boxes, tf.range(batch_size), [12, 12])
            img_tar_crop = tf.image.crop_and_resize(img_tar_f2, boxes, tf.range(batch_size), [12, 12])
            crop_diff_flow2 = shifted_crop - img_tar_crop

            ####F3
            f3 = self.interpolation(mot_flow_3, trainable=True, reuse_core=reuse_core)

            flow3 = tf.reshape(f3, [batch_size, 1, 1, 2])
            flow3_state = tf.tile(flow3, [1, 8, 8, 1])
            #flow3 = tf.tile(flow3, [1, 240 + 64, 320 + 64, 1])
            flow3 = tf.tile(flow3, [1, 288 + 64, 352 + 64, 1])

            #grid_x, grid_y = meshgrid(240 + 64, 320 + 64)
            grid_x, grid_y = meshgrid(288 + 64, 352 + 64)

            grid_x = tf.tile(grid_x, [batch_size, 1, 1])  # batch_size = 100
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])  # batch_size = 100

            coor_x = grid_x + flow3[:, :, :, 0]
            coor_y = grid_y + flow3[:, :, :, 1]

            shifted = bilinear_interp(img_last_f3, coor_x, coor_y, 'interpolate')
            
            """
            x1 = (x - 1. * 6. + 32.) / (240. + 64.)
            x2 = (x + 1. * 6. + 32.) / (240. + 64.)
            y1 = (y - 1. * 6. + 32.) / (320. + 64.)
            y2 = (y + 1. * 6. + 32.) / (320. + 64.)
            """
            x1 = (x - 1. * 6. + 32.) / (288. + 64.)
            x2 = (x + 1. * 6. + 32.) / (288. + 64.)
            y1 = (y - 1. * 6. + 32.) / (352. + 64.)
            y2 = (y + 1. * 6. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)

            shifted_crop = tf.image.crop_and_resize(shifted, boxes, tf.range(batch_size), [12, 12])
            img_tar_crop = tf.image.crop_and_resize(img_tar_f3, boxes, tf.range(batch_size), [12, 12])
            crop_diff_flow3 = shifted_crop - img_tar_crop

            ####FLOW4
            extrapolation_1 = tf.layers.conv2d(inputs= mot_state_1, filters=  8, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_1")
            extrapolation_1 = tf.contrib.layers.max_pool2d(extrapolation_1, [2, 2]) #32

            extrapolation_2 = tf.layers.conv2d(inputs= extrapolation_1, filters=  16, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_2")

            extrapolation_3 = tf.layers.conv2d(inputs= extrapolation_2, filters=  32, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_3")
            extrapolation_3 = tf.contrib.layers.max_pool2d(extrapolation_3, [2, 2]) #16

            extrapolation_4 = tf.layers.conv2d(inputs= extrapolation_3, filters=  64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_4")

            extrapolation_5 = tf.layers.conv2d(inputs= extrapolation_4, filters=  128, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_5")
            extrapolation_5 = tf.contrib.layers.max_pool2d(extrapolation_5, [2, 2]) #8

            extrapolation_6 = tf.layers.conv2d(inputs=extrapolation_5, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_6")
            extrapolation_7 = tf.layers.conv2d(inputs=extrapolation_6, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_7")
            extrapolation_8 = tf.layers.conv2d(inputs=extrapolation_7, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_8")
            extrapolation_9 = tf.layers.conv2d(inputs=extrapolation_8, filters=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_9")
            extrapolation_10 = tf.layers.conv2d(inputs=extrapolation_9, filters=8, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_10")

            extrapolation_10 = tf.concat(values = [extrapolation_10, flow1_state, flow2_state, flow3_state], axis = -1)

            extrapolation_11 = tf.layers.conv2d(inputs=extrapolation_10, filters=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME", trainable=trainable, reuse=reuse_model, name="extrapolation_11")
            mot = tf.reduce_mean(extrapolation_11, [1, 2])
            mot_out = tf.reshape(mot,[batch_size, 2])

            flow = tf.reshape(mot,[batch_size, 1, 1, 2])
            #flow = tf.tile(flow, [1, 240 + 64, 320 + 64, 1])
            flow = tf.tile(flow, [1, 288 + 64, 352 + 64, 1])

            #grid_x, grid_y = meshgrid(240 + 64, 320 + 64)
            grid_x, grid_y = meshgrid(288 + 64, 352 + 64)

            grid_x = tf.tile(grid_x, [batch_size, 1, 1])  # batch_size = 100
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])  # batch_size = 100

            coor_x = grid_x + flow[:, :, :, 0]
            coor_y = grid_y + flow[:, :, :, 1]

            shifted = bilinear_interp(img_last, coor_x, coor_y, 'interpolate')
            
            """
            x1 = (x - 1. * 6. + 32.) / (240. + 64.)
            x2 = (x + 1. * 6. + 32.) / (240. + 64.)
            y1 = (y - 1. * 6. + 32.) / (320. + 64.)
            y2 = (y + 1. * 6. + 32.) / (320. + 64.)
            """
            x1 = (x - 1. * 6. + 32.) / (288. + 64.)
            x2 = (x + 1. * 6. + 32.) / (288. + 64.)
            y1 = (y - 1. * 6. + 32.) / (352. + 64.)
            y2 = (y + 1. * 6. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)

            shifted_crop = tf.image.crop_and_resize(shifted, boxes, tf.range(batch_size), [12, 12])
            img_tar_crop = tf.image.crop_and_resize(img_tar, boxes, tf.range(batch_size), [12, 12])
            crop_diff = shifted_crop - img_tar_crop

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        return mot_out, crop_diff, crop_diff_flow1, crop_diff_flow2, crop_diff_flow3, params


    def region_adaptive_model(self, pos, mot, state_pad, batch_size, scope, trainable, reuse_model, reuse_core):
        with tf.variable_scope(scope):

            StatePad = tf.slice(state_pad, [0, 0, 0, 6], [-1, -1, -1, 6])
            last_img = tf.slice(state_pad, [0, 0, 0, 9], [-1, -1, -1, 3])
            last_last_img = tf.slice(state_pad, [0, 0, 0, 6], [-1, -1, -1, 3])
            

            x = tf.slice(pos, [0, 0], [-1, 1]) * 4.
            y = tf.slice(pos, [0, 1], [-1, 1]) * 4.

            u = tf.slice(mot, [0, 0], [-1, 1])  #y
            v = tf.slice(mot, [0, 1], [-1, 1])  #x
            
            """
            x1 = (x + v - 1. * 32. + 32.) / (240. + 64.)
            x2 = (x + v + 1. * 32. + 32.) / (240. + 64.)
            y1 = (y + u - 1. * 32. + 32.) / (320. + 64.)
            y2 = (y + u + 1. * 32. + 32.) / (320. + 64.)
            """
            x1 = (x + v - 1. * 32. + 32.) / (288. + 64.)
            x2 = (x + v + 1. * 32. + 32.) / (288. + 64.)
            y1 = (y + u - 1. * 32. + 32.) / (352. + 64.)
            y2 = (y + u + 1. * 32. + 32.) / (352. + 64.)
            #"""
            
            boxes = tf.concat(values=[x1, y1, x2, y2], axis=1)
            StatePad_crop = tf.image.crop_and_resize(StatePad, boxes, tf.range(batch_size), [64, 64])

            flow = self.interpolation(StatePad_crop, trainable=True, reuse_core=reuse_core)
            flow = tf.reshape(flow, [batch_size, 1, 1, 2])
            #flow = tf.tile(flow, [1, 240 + 64, 320 + 64, 1])
            flow = tf.tile(flow, [1, 288 + 64, 352 + 64, 1])

            #grid_x, grid_y = meshgrid(240 + 64, 320 + 64)
            grid_x, grid_y = meshgrid(288 + 64, 352 + 64)

            grid_x = tf.tile(grid_x, [batch_size, 1, 1])  # batch_size = 100
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])  # batch_size = 100

            coor_x = grid_x + flow[:, :, :, 0]
            coor_y = grid_y + flow[:, :, :, 1]

            shifted = bilinear_interp(last_last_img, coor_x, coor_y, 'interpolate')
            
            shifted_crop = tf.image.crop_and_resize(shifted, boxes, tf.range(batch_size), [64, 64])
            img_tar_crop = tf.image.crop_and_resize(last_img, boxes, tf.range(batch_size), [64, 64])
            crop_diff = tf.abs(shifted_crop - img_tar_crop)

            region_adaptive_1 = tf.layers.conv2d(inputs=crop_diff, filters=8, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="region_adaptive_1")
            region_adaptive_1 = tf.contrib.layers.max_pool2d(region_adaptive_1, [2, 2])  # 32

            region_adaptive_2 = tf.layers.conv2d(inputs=region_adaptive_1, filters=16, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="region_adaptive_2")

            region_adaptive_3 = tf.layers.conv2d(inputs=region_adaptive_2, filters=32, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="region_adaptive_3")
            region_adaptive_3 = tf.contrib.layers.max_pool2d(region_adaptive_3, [2, 2])  # 16

            region_adaptive_4 = tf.layers.conv2d(inputs=region_adaptive_3, filters=64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="region_adaptive_4")

            region_adaptive_5 = tf.layers.conv2d(inputs=region_adaptive_4, filters=128, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding="SAME", trainable=trainable, reuse=reuse_model, name="region_adaptive_5")
            region_adaptive_5 = tf.contrib.layers.max_pool2d(region_adaptive_5, [2, 2])  # 8

            shape = region_adaptive_5.get_shape().as_list()
            region_adaptive_state_2 = tf.reshape(region_adaptive_5, [-1, reduce(lambda x, y: x * y, shape[1:])])
            region_adaptive_6 = tf.layers.dense(region_adaptive_state_2, 512, activation=tf.nn.relu, trainable=trainable, reuse=reuse_model, name="region_adaptive_6")
            region_adaptive_7 = tf.layers.dense(region_adaptive_6, 256, activation=tf.nn.relu, trainable=trainable, reuse=reuse_model, name="region_adaptive_7")
            region_adaptive_8 = tf.layers.dense(region_adaptive_7, 64, activation=tf.nn.relu, trainable=trainable, reuse=reuse_model, name="region_adaptive_8")
            region_adaptive = tf.layers.dense(region_adaptive_8, 7, activation=tf.nn.relu, trainable=trainable, reuse=reuse_model, name="region_adaptive_9")

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        return region_adaptive, params
    
    
    def greedy_pos(self, height, width):
          
        # input
        self.greedy_mask = tf.placeholder(tf.float32, shape=(None, height, width, 1))
        self.residual_frame = tf.placeholder(tf.float32, shape=(None, height, width, 1))
        mask_residual = tf.multiply(self.residual_frame, 1 - self.greedy_mask)
        
        # convolution
        kernel_size = 16 + 1
        ukernel = np.ones((kernel_size, kernel_size, 1, 1))/(kernel_size**2) # uniform filter
        kernel = tf.constant(value=ukernel, dtype=tf.float32)
        mean_residual = tf.nn.conv2d(mask_residual, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # find max residual part index
        shape = mean_residual.get_shape().as_list()
        mean_residual = tf.reshape(mean_residual, [-1, reduce(lambda x, y: x * y, shape[1:])])
        
        greedy_index = tf.expand_dims(tf.argmax(mean_residual, axis = -1), 1)
        greedy_height = tf.div(greedy_index, width)
        greedy_width = tf.floormod(greedy_index, width)
        
        return greedy_height, greedy_width
    
    
    # --------------------------
    # RL update function
    # --------------------------
    def update_joint(self, state_seq, state_pad_seq, mask_seq, pos_seq, mot_seq, picked_pos_label_seq, picked_order_label_seq, target_seq, batch_size):
        
        # model format
        feed_dict = {self.state_seq[i]: state_seq[i] for i in range(self.training_process_num)}
        feed_dict.update({self.state_pad_seq[i]: state_pad_seq[i] for i in range(self.training_process_num)})
        feed_dict.update({self.mask_seq[i]: mask_seq[i] for i in range(self.training_process_num)})
        
        feed_dict.update({self.region_adaptive_pos_seq[i]: pos_seq[i] for i in range(self.training_process_num)})
        feed_dict.update({self.region_adaptive_mot_seq[i]: mot_seq[i] for i in range(self.training_process_num)})
        
        feed_dict.update({self.picked_pos_label_seq[i]: picked_pos_label_seq[i] for i in range(self.training_process_num)})
        feed_dict.update({self.picked_order_label_seq[i]: picked_order_label_seq[i] for i in range(self.training_process_num)})
        feed_dict.update({self.target_seq[i]: target_seq[i] for i in range(self.training_process_num)})
        
        feed_dict.update({self.batch_size_train: batch_size})
        
        # update pi
        _, loss, order_predict, order_pro, pos_pro = self.sess.run([self.train_op_j, self.jloss_seq, self.train_picked_order, self.train_picked_order_prob, self.train_picked_pos_prob], feed_dict)
        
        # update old pi
        self.sess.run(self.update_oldpi_position)
        self.sess.run(self.update_oldpi_region_adaptive)
        #self.sess.run(self.update_oldpi_motion)

        return loss, order_predict, order_pro, pos_pro


    # --------------------------
    # Motion update function
    # --------------------------
    def update_motion(self, state_pad_seq, pos_seq, batch_size):
        feed_dict = {self.state_pad_seq[i]: state_pad_seq[i] for i in range(self.training_process_num)}
        feed_dict.update({self.mot_state_seq[i]: pos_seq[i] for i in range(self.training_process_num)})
        feed_dict.update({self.batch_size_train: batch_size})

        _, loss = self.sess.run([self.train_op_m, self.mloss_seq], feed_dict)
        self.sess.run(self.update_oldpi_motion)

        return loss



    # --------------------------
    # RL play function
    # --------------------------
    def play_position(self, state, mask, random_choice, batch_size):
        
        index, height, width = [], [], []
        if random_choice:  # random choose
            index, height, width = self.sess.run([self.pos_choice_random, self.pos_height_random, self.pos_width_random], 
                feed_dict={self.state: state, self.mask: mask})
        else:              # always choose max
            index, height, width = self.sess.run([self.pos_choice_max, self.pos_height_max, self.pos_width_max], 
                feed_dict={self.state: state, self.mask: mask})
        
        return index[:,0], height, width
    
    
    def play_motion(self, state_pad, mot_state ,random_choice, batch_size):
        
        return self.sess.run(self.play_mot, feed_dict={
            self.state_pad: state_pad, self.mot_state: mot_state, self.batch_size_play: batch_size})


    def play_mot_var(self, state_pad, mot_state, var_decay, random_choice, batch_size):

        motion = self.sess.run(self.play_mot, feed_dict={self.state_pad: state_pad, self.mot_state: mot_state, self.batch_size_play: batch_size})

        if random_choice:  # random choose position

            #LA_loss = np.linalg.norm(motion - var_ref, axis=1)
            #var = np.expand_dims(var_decay * np.clip(LA_loss / 4., 0, 2) + 0.1, 1)
            var = np.expand_dims(var_decay * 0.1, 1)
            var_2D = np.repeat(var, 2, axis=1)

            return motion + np.multiply(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], (batch_size)), var_2D)

        else:  # always choose max position
            return motion



    def play_region_adaptive(self, region_adaptive_pos, region_adaptive_mot, state_pad ,random_choice, batch_size):
        
        index, order = [], []
        if random_choice:  # random choose
            index, order = self.sess.run([self.region_adaptive_choice_random, self.order_choice_random], feed_dict={self.region_adaptive_pos: region_adaptive_pos, 
                self.region_adaptive_mot: region_adaptive_mot, self.state_pad: state_pad, self.batch_size_play: batch_size})
        else:              # always choose max
            index, order = self.sess.run([self.region_adaptive_choice_max, self.order_choice_max], feed_dict={self.region_adaptive_pos: region_adaptive_pos, 
                self.region_adaptive_mot: region_adaptive_mot, self.state_pad: state_pad, self.batch_size_play: batch_size})
        
        return index[:,0], order
    
    # --------------------------
    # pretrain part
    # --------------------------
    def greedy_method(self, residual, mask, height, width, downsample, batch_size):
        
        # get max residual position
        ori_x, ori_y = self.sess.run([self.greedy_height, self.greedy_width], feed_dict={
            self.residual_frame: residual, self.greedy_mask: mask})
        
        # downsample
        index_map, down_x, down_y = [], [], []
        down_height, down_width = int(height/downsample), int(width/downsample)
        for i in range(batch_size):
            down_x.append(int(ori_x[i]/downsample))
            down_y.append(int(ori_y[i]/downsample))
            single_map = np.zeros([down_height*down_width])
            label = down_x[i]*down_width + down_y[i]
            single_map[label - 1] = 1
            index_map.append(single_map)
        
        return index_map, down_x, down_y
    
    """
    def pretrain_mot(self, state, state_pad, mot_state, context_frame, target_frame):
        _, loss, mot = self.sess.run([self.optimize_pretrain_mot, self.pre_train_cost_mot, self.play_mot], feed_dict={
            self.state: state,
            self.state_pad: state_pad,
            self.mot_state: mot_state,
            self.context_frame: context_frame,
            self.target_frame: target_frame
        })
        return loss, mot
    
    
    def eva_pretrain_mot(self, state, state_pad, mot_state, context_frame, target_frame):
        loss, mot = self.sess.run([self.pre_train_cost_mot, self.play_mot], feed_dict={
            self.state: state,
            self.state_pad: state_pad,
            self.mot_state: mot_state,
            self.context_frame: context_frame,
            self.target_frame: target_frame
        })
        return loss, mot
    """
    
    # --------------------------
    # save model function
    # --------------------------
    def get_session(self, sess):
        session = sess
        while type(session).__name__ != 'Session':
            #pylint: disable=W0212
            session = session._sess
        return session
    
    
    def restore_weight(self, project, task, label=None):
        
        # initial model name
        #model_name = []
        if label is None:
            model_name = "PolicyEstimator_%s" % task
        else:
            model_name = "PolicyEstimator_%s" % task + str(label)
        
        # restore pi
        try:
            # restore pi
            if task == "position":
                self.saver_position.restore(self.sess, tf.train.latest_checkpoint('./%s/%s'%(project, model_name)))
            elif task == "motion":
                self.saver_motion.restore(self.sess, tf.train.latest_checkpoint('./%s/%s'%(project, model_name)))
            elif task == "region_adaptive":
                self.saver_region_adaptive.restore(self.sess, tf.train.latest_checkpoint('./%s/%s'%(project, model_name)))

            print('Restore PolicyEstimator_%s Weights from %s'%(task, project))
        except:
            print('Initialize PolicyEstimator_%s Weights' % task)
        
    
    def save_weight(self, project, task, label=None):
        
        # initial model name
        #model_name = []
        if label is None:
            model_name = "PolicyEstimator_%s" % task
        else:
            model_name = "PolicyEstimator_%s" % task + str(label)
        
        # build epi-folder
        path =  "./%s/%s"%(project, model_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        # save
        if task == "position":
            self.saver_position.save(self.get_session(self.sess), './%s/%s/model_weight'%(project, model_name))
        elif task == "motion":
            self.saver_motion.save(self.get_session(self.sess), './%s/%s/model_weight'%(project, model_name))
        elif task == "region_adaptive":
            self.saver_region_adaptive.save(self.get_session(self.sess), './%s/%s/model_weight'%(project, model_name))
    
    
    # --------------------------
    # others..
    # --------------------------
    def get_training_info(self):
        return self.training_process_num
    
    
