import numpy as np
import tensorflow as tf

import time

# import SSIM func [sudo pip install pyssim ]
import ssim
from PIL import Image

from spt_tf3 import transformer
from visualization_rgb import Visualization

class Env_toolbox(object):
    def __init__(self, steps, multi_frame, height, width):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.InteractiveSession(config=config)
            
            self.steps  = steps
            self.multi_frame = multi_frame
            self.height = height
            self.width  = width
            self.seq_content_r = np.empty((self.multi_frame, self.height*self.width*self.steps), dtype=np.float32)
            self.seq_content_g = np.empty((self.multi_frame, self.height*self.width*self.steps), dtype=np.float32)
            self.seq_content_b = np.empty((self.multi_frame, self.height*self.width*self.steps), dtype=np.float32)
            
            # initial MC_Accelerator step by step
            self.obmc_toolbox = MC_Accelerator(self.sess, steps, multi_frame, height, width)
            
            # initial Reward_Estimator
            self.reward_toolbox = Delta_Reward_Estimator(self.sess, steps, multi_frame, height, width, version="LA")
            
            # initial Frame_Transformer
            self.shift_toolbox = Frame_Transformer(self.sess, multi_frame, height, width)
            
            # initial state
            self.game_over()
            
            #initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())
    
    def game_over(self):
        self.process_step = 0
        self.critical_point = np.ones((self.multi_frame, self.steps, 2))*100000 # just initial using a big number
        self.critical_order = np.ones((self.multi_frame, self.steps))*1.5
    
    def do_obmc(self, position_update, motion_update, order_update, current_frame):
        
        # get shift map
        current_frame = np.reshape(current_frame, (self.multi_frame, self.height, self.width, 3))
        shift_map_update_r = self.shift_toolbox.get_transformed_frame(np.expand_dims(current_frame[:,:,:,0],3), motion_update)
        shift_map_update_g = self.shift_toolbox.get_transformed_frame(np.expand_dims(current_frame[:,:,:,1],3), motion_update)
        shift_map_update_b = self.shift_toolbox.get_transformed_frame(np.expand_dims(current_frame[:,:,:,2],3), motion_update)
        
        # update state
        self.critical_point[:,self.process_step,:] = position_update
        self.critical_order[:,self.process_step] = order_update[:,0]
        self.seq_content_r[:, self.height*self.width*self.process_step: self.height*self.width*(self.process_step+1)] = shift_map_update_r
        self.seq_content_g[:, self.height*self.width*self.process_step: self.height*self.width*(self.process_step+1)] = shift_map_update_g
        self.seq_content_b[:, self.height*self.width*self.process_step: self.height*self.width*(self.process_step+1)] = shift_map_update_b
        
        self.process_step += 1
        step_num = self.process_step
        if step_num < 2:
            step_num = 2
        
        if self.process_step < self.steps:
            return np.zeros((self.height, self.width, 3))
        else:
            # get rgb value
            mcframe_r = self.obmc_toolbox.do_obmc(self.critical_point[:,0:step_num,:], self.critical_order[:,0:self.steps], self.seq_content_r)
            mcframe_g = self.obmc_toolbox.do_obmc(self.critical_point[:,0:step_num,:], self.critical_order[:,0:self.steps], self.seq_content_g)
            mcframe_b = self.obmc_toolbox.do_obmc(self.critical_point[:,0:step_num,:], self.critical_order[:,0:self.steps], self.seq_content_b)
            
            # get mcframe
            mcframe_rgb = np.zeros((self.multi_frame, self.height, self.width, 3))
            mcframe_rgb[:,:,:,0] = mcframe_r
            mcframe_rgb[:,:,:,1] = mcframe_g
            mcframe_rgb[:,:,:,2] = mcframe_b
            
            return mcframe_rgb
    
    def get_delta_reward(self, mc_frame, current_frame, target_frame):
        
        self.game_over() # reset game
        return self.reward_toolbox.get_delta_reward(mc_frame, current_frame, target_frame)
    
    def get_last_reward(self, mc_frame, current_frame, target_frame):
        
        self.game_over() # reset game
        return self.reward_toolbox.get_last_reward(mc_frame, current_frame, target_frame)
    
    def update_test_info(self, predict_frame, current_frame, target_frame, task):
        
        # reshape
        predict_frame = np.reshape(predict_frame, (self.height, self.width, 3))*255.
        current_frame = np.reshape(current_frame, (self.height, self.width, 3))*255.
        target_frame = np.reshape(target_frame, (self.height, self.width, 3))*255.
        predict_frame = predict_frame.astype("uint8")
        current_frame = current_frame.astype("uint8")
        target_frame = target_frame.astype("uint8")
        
        # remove boundry
        cutwidth = 0
        predict_frame = predict_frame[:,cutwidth:352,:]
        current_frame = current_frame[:,cutwidth:352,:]
        target_frame = target_frame[:,cutwidth:352,:]
        
        # get mse
        MSE = ((predict_frame/255. - target_frame/255.) ** 2).mean(axis=None)
        
        # get PSNR
        PSNR = 10.*np.log10(1./MSE)
        
        # get reward
        total_reward = 0#self.reward_toolbox.get_last_reward(predict_frame, current_frame, target_frame)
        REWARD = np.sum(total_reward)
        
        # get SSIM
        SSIM = ssim.compute_ssim(Image.fromarray(target_frame),Image.fromarray(predict_frame))
        
        # task part
        if task == "Save_Copy":
            self.test_visual.update_copylast_info(MSE, PSNR, REWARD, SSIM)
        elif task == "Save_Test":
            self.test_visual.save_image(predict_frame, current_frame, target_frame)    # save image
            self.test_visual.update_pretrain_and_predict_info(MSE, PSNR, REWARD, SSIM)
        elif task == "Don't_Save":
            return MSE, PSNR, REWARD, SSIM
        else:
            input("Task Error !!")
        
        return
        
    def build_visualization(self, save_folder, test_num):
        if self.multi_frame == 1: # just test case can use this class
            self.test_visual = Visualization(self.height, self.width, save_folder, "test", test_num)
        else:
            input("Build Visualization Error !!")


class MC_Accelerator(object):
    def __init__(self, sess, steps, multi_frame, height, width):
        
        self.sess = sess
        self.multi_frame = multi_frame
        
        # -----------------------------------------------
        # Preprocess: get image format and position map
        # -----------------------------------------------
        self.height = height
        self.width = width
        self.total_pixel = height*width
        
        # get position label
        self.position_label = np.array(range(self.total_pixel), dtype = np.float32)
        
        # get position map
        self.position_map = []
        for i in range(height):
            for j in range(width):
                self.position_map.append(np.array([i, j], dtype = np.float32))
        self.position_map = np.array(self.position_map)
        
        # get search_point
        reshape_search_point = np.reshape(np.tile(self.position_map, [1, steps]), [-1, steps, 2])
        
        # -----------------------------------------------
        # Part I: get classification map and weight
        # -----------------------------------------------
        # input
        self.critical_point = [tf.placeholder(tf.float32, shape=(steps, 2)) for i in range(multi_frame)]
        self.critical_order = [tf.placeholder(tf.float32, shape=(steps)) for i in range(multi_frame)]
        
        # output
        self.weight1, self.weight2 = [], []
        self.first_one_index_group, self.second_one_index_group = [], []
        
        # get distance -> find first and second index -> get weight
        for i in range(multi_frame):
            
            # get first index distance
            position_difference = reshape_search_point - self.critical_point[i]
            distance_square = tf.multiply(position_difference,position_difference)
            position_distance1 = tf.pow(tf.reduce_sum(distance_square, 2), self.critical_order[i])
            
            # get first and second index using first and second distance
            first_one_index = tf.argmin(position_distance1, axis = -1)
            first_onehot = tf.one_hot(indices = first_one_index, depth = steps, on_value = 1., off_value = 0.)
            position_distance2 = position_distance1 + 1000000.*first_onehot # just add a big number in "first_one_index" position
            second_one_index = tf.argmin(position_distance2, axis = -1)
        
            # get weight ratio
            first_one_value  = 1./(tf.reduce_min(position_distance1, axis = -1)+0.00001)
            second_one_value = 1./(tf.reduce_min(position_distance2, axis = -1)+0.00001)
            
            # get final weight
            weight_sum = first_one_value + second_one_value
            self.weight1.append( first_one_value/weight_sum)
            self.weight2.append(second_one_value/weight_sum)
            
            # update index group (for index classification)
            self.first_one_index_group.append(first_one_index*self.total_pixel + self.position_label)
            self.second_one_index_group.append(second_one_index*self.total_pixel + self.position_label)
        
    
    def get_parameter_part1(self, critical_point, critical_order):
        
        temp = np.split(critical_point, self.multi_frame)
        feed_dict = {self.critical_point[i]: np.squeeze(temp[i], axis=0) for i in range(self.multi_frame)}
        temp = np.split(critical_order, self.multi_frame)
        feed_dict.update({self.critical_order[i]: np.squeeze(temp[i], axis=0) for i in range(self.multi_frame)})
        return self.sess.run([self.weight1, self.weight2, self.first_one_index_group, self.second_one_index_group], feed_dict)
    
    
    def get_parameter_part2(self, index_group1, index_group2, seq_map):
        
        map1, map2 = [], []
        for frame_label in range(self.multi_frame):
            
            step_i1 = np.take(seq_map[frame_label,:], index_group1[frame_label])
            step_i2 = np.take(seq_map[frame_label,:], index_group2[frame_label])
            
            # update index
            map1.append(step_i1)
            map2.append(step_i2)
        
        return map1, map2
    
    
    def do_obmc(self, critical_point, critical_order, seq_content):
        
        # Step1: get weight and its index group
        w1, w2, ig1, ig2 = self.get_parameter_part1(np.array(critical_point), np.array(critical_order))
        # Step2: generate first_map and second_map using "tf.gather"
        first_map, second_map = self.get_parameter_part2(ig1, ig2, seq_content)
        # Step3: weighted sum operation
        self.weighted_sum_map = np.multiply(first_map, w1) + np.multiply(second_map, w2)
        
        return np.reshape(self.weighted_sum_map, (self.multi_frame, self.height, self.width))
    

class Frame_Transformer(object):
    def __init__(self, sess, multi_frame, height, width):
        
        self.sess = sess
        
        # input
        self.frame  = tf.placeholder(tf.float32, shape=(None, height, width, 1))
        self.motion = tf.placeholder(tf.float32, shape=(None, 6))
        
        # transform
        self.transformed_frame = transformer(self.frame, self.motion, (height, width))
    
    def get_transformed_frame(self, frame, motion):
        feed_dict = {self.frame: frame, self.motion: self.reformat(motion)}
        return self.sess.run(self.transformed_frame, feed_dict)
    
    def reformat(self, motion):
        
        # get seq_num
        seq_num = (np.shape(motion))[0]
        
        # get reformat motion
        reformat_motion = []
        for i in range(seq_num):
            reformat_motion.append([1.0, 0.0, motion[i][0], 0.0, 1.0, motion[i][1]])
        
        return reformat_motion

        
class Delta_Reward_Estimator(object):
    def __init__(self, sess, steps, multi_frame, height, width, version):
        
        self.sess = sess
        self.steps = steps
        self.multi_frame = multi_frame
        self.total_pixel = height*width*3
        self.version = version
        
        # input
        self.frame1 = tf.placeholder(tf.float32, shape=(None, self.total_pixel))
        self.frame2 = tf.placeholder(tf.float32, shape=(None, self.total_pixel))
    
        # get sum of square error
        loss = self.frame1/255. - self.frame2/255.
        
        # get LA
        square_error = tf.multiply(loss, loss)
        sum_square_error = tf.reduce_sum(square_error, axis = -1)
        self.LA = -1.*tf.sqrt(sum_square_error)
        
        # get L1
        abs_error = tf.abs(loss)
        self.L1 = -1.*tf.reduce_sum(abs_error, axis = -1)
    
    
    def get_delta_reward(self, seq_mc, current_frame, target_frame):
        
        # output
        total_reward = np.zeros((self.multi_frame, self.steps))
        
        # normalize parameter
        loss_current_target = []
        if self.version == "LA":
            loss_current_target = self.get_LA_loss(current_frame, target_frame)
        elif self.version == "L1":
            loss_current_target = self.get_L1_loss(current_frame, target_frame)
            
        # memory
        last_step_mc_loss = loss_current_target
        
        # get reward for every step
        for step_label in range(0, self.steps):
            
            # get new mc lss
            new_mc_loss = []
            if self.version == "LA":
                new_mc_loss = self.get_LA_loss(seq_mc[step_label], target_frame)
            elif self.version == "L1":
                new_mc_loss = self.get_L1_loss(seq_mc[step_label], target_frame)
            
            # get reward (delta loss)
            total_reward[:, step_label] = np.divide(last_step_mc_loss - new_mc_loss, loss_current_target)
            #total_reward[:, step_label] = np.divide(last_step_mc_loss - new_mc_loss, last_step_mc_loss)
            
            # update last_step_mc_loss
            last_step_mc_loss = new_mc_loss
        
        return total_reward
    
    def get_last_reward(self, predict_frame, current_frame, target_frame):
        
        # output
        total_reward = np.zeros((self.multi_frame, self.steps))
        
        # normalize parameter
        loss_current_target = []
        if self.version == "LA":
            loss_current_target = self.get_LA_loss(current_frame, target_frame)
        elif self.version == "L1":
            loss_current_target = self.get_L1_loss(current_frame, target_frame)
            
        # get new mc lss
        new_mc_loss = []
        if self.version == "LA":
            new_mc_loss = self.get_LA_loss(predict_frame, target_frame)
        elif self.version == "L1":
            new_mc_loss = self.get_L1_loss(predict_frame, target_frame)
        
        # get reward (delta loss)
        #total_reward[:, self.steps-1] = np.divide(loss_current_target - new_mc_loss, loss_current_target)    # current <-> mc, normalize
        #total_reward[:, self.steps-1] = new_mc_loss - loss_current_target                                     # current <-> mc, ordinary
        total_reward[:, self.steps-1] = new_mc_loss                                                          # mc <-> target, ordinary
        
        return total_reward
    
    def get_LA_loss(self, frame1, frame2):
        
        # reshape
        frame1 = np.reshape(frame1, (self.multi_frame, self.total_pixel))
        frame2 = np.reshape(frame2, (self.multi_frame, self.total_pixel))
        
        # get LA loss
        loss = self.sess.run(self.LA, feed_dict={self.frame1: frame1, self.frame2: frame2})
        
        return np.array(loss)
    
    def get_L1_loss(self, frame1, frame2):
        
        # reshape
        frame1 = np.reshape(frame1, (self.multi_frame, self.total_pixel))
        frame2 = np.reshape(frame2, (self.multi_frame, self.total_pixel))
        
        # get LA loss
        loss = self.sess.run(self.L1, feed_dict={self.frame1: frame1, self.frame2: frame2})
        
        return np.array(loss)
        
        
        
    
    
    