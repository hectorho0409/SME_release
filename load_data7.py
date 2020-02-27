import numpy as np
import scipy.io as scio                     # load flow data
from scipy.misc import imread, imsave       # import "imread", "imsave" function
import pickle as cp                         # load dataset
import os                                   # build folder

ROOT_TO_CIF_DIR = "../pretrain_cif/cif_rgb_dataset/"

class Load_cif_test(object):
    def __init__(self, height, width, downsample, batch_size, data_route):
        
        self.dataset_name = "news"
        self.height = height
        self.width = width
        self.down_height = int(height/downsample)
        self.down_width = int(width/downsample)
        self.downsample = downsample
        self.batch_size = batch_size
        self.mask_size = 4  # mask radius
        #self.dataset_list = np.load(data_route)  # dataset list
        self.reset()
    
    
    def reset(self):
            
        # model imput
        self.image_0, self.image_1, self.image_2, self.image_3, self.mask = [], [], [], [], []
        self.image_0_pad, self.image_1_pad, self.image_2_pad, self.image_3_pad, self.image_4_pad = [], [], [], [], []
        self.base_frame, self.base_frame_rgb = [], []
        
        # greedy method
        self.greedy_mask = []
        
        # visualization input
        self.reference_frame, self.object_frame, self.mc_seq, self.mc_seq_rgb = [], [], [], []
        self.reference_frame_rgb, self.object_frame_rgb = [], []
        self.position_map = []
        
        # load data process
        self.step_label = 0
    
    
    def game_over(self):
            
        # save old data
        reference_frame, object_frame = self.reference_frame_rgb, self.object_frame_rgb
        mc_seq, mc_seq_rgb = self.mc_seq, self.mc_seq_rgb
        position_map = self.position_map
        
        # reset
        self.reset()
        
        return reference_frame, object_frame, mc_seq, mc_seq_rgb, position_map
    
    def get_model_input1(self, search_label,  continue_time):
        
        # generate state_image
        jump = 1
        state, state_pad, target_frame, data_name = [], [], [], search_label + continue_time + 3*jump
        for batch_label in range(self.batch_size):
            
            # load dataset
            pred_mc = []
            if self.step_label == 0:
                # model imput
                self.image_0.append(self.load_image(data_name, -3*jump, continue_time))
                self.image_1.append(self.load_image(data_name, -2*jump, continue_time))
                self.image_2.append(self.load_image(data_name, -1*jump, continue_time))
                self.image_3.append(self.load_image(data_name,  0*jump, continue_time))
                self.image_0_pad.append(np.pad(self.load_image_rgb(data_name, -3*jump, continue_time), ((32, 32), (32, 32), (0, 0)), 'edge'))
                self.image_1_pad.append(np.pad(self.load_image_rgb(data_name, -2*jump, continue_time), ((32, 32), (32, 32), (0, 0)), 'edge'))
                self.image_2_pad.append(np.pad(self.load_image_rgb(data_name, -1*jump, continue_time), ((32, 32), (32, 32), (0, 0)), 'edge'))
                self.image_3_pad.append(np.pad(self.load_image_rgb(data_name, -0*jump, continue_time), ((32, 32), (32, 32), (0, 0)), 'edge'))
                self.image_4_pad.append(np.pad(self.load_image_rgb(data_name,  0*jump, continue_time), ((32, 32), (32, 32), (0, 0)), 'edge'))  # useless in test case
                self.mask.append(np.zeros((self.down_height, self.down_width, 1)))
                pred_mc = self.image_3[batch_label]
                
                # env input
                self.base_frame.append(self.load_image(data_name, 0*jump, continue_time))
                self.base_frame_rgb.append(self.load_image_rgb(data_name, 0*jump, continue_time))
                
                # visualization input (gray)
                self.reference_frame.append(self.load_image(search_label + 3*jump, 0*jump, 0))
                self.object_frame.append(self.load_image(data_name, 1*jump, 0))
                
                # visualization input (rgb)
                self.reference_frame_rgb.append(self.load_image_rgb(search_label + 3*jump, 0*jump, 0))
                self.object_frame_rgb.append(self.load_image_rgb(data_name, 1*jump, 0))
                
                # visualization input (pisition map)
                self.position_map.append(np.zeros((self.height, self.width, 2)))
                
            else:
                # just pred_mc need to update
                pred_mc = np.array(np.expand_dims(self.mc_seq[self.step_label-1][batch_label,:,:], 2))
            
            # update state (position part)
            s = []
            s.append(abs(self.image_1[batch_label] - self.image_0[batch_label]))
            s.append(abs(self.image_2[batch_label] - self.image_0[batch_label]))
            s.append(abs(self.image_3[batch_label] - self.image_0[batch_label]))
            s.append(abs(pred_mc - self.image_0[batch_label]))
            s = np.transpose(s, (3, 1, 2, 0))
            state.append(s[0])
            
            # update state_pad (motion part)
            s = []
            s.append(self.image_0_pad[batch_label])
            s.append(self.image_1_pad[batch_label])
            s.append(self.image_2_pad[batch_label])
            s.append(self.image_3_pad[batch_label])
            s.append(self.image_4_pad[batch_label])
            s = np.concatenate(s, -1)
            state_pad.append(s)
            
            # update target (just context= 1)
            s = []
            s.append(self.object_frame[batch_label])
            target_frame.append(s)
        
        # update step process
        self.step_label += 1
        
        return state, state_pad, np.array(self.mask), target_frame
    
    
    def get_model_input2(self, pos_height, pos_width):
        
        down_pos = []
        for batch_label in range(self.batch_size):
            
            # update downsample position
            down_pos.append(np.concatenate([pos_height[batch_label], pos_width[batch_label]], -1))
        
        return down_pos, None
    
    
    def get_env_input(self, pos_height, pos_width, action_mot=None):
        
        # initial obmc input
        position_batch, motion_batch = [], []
        
        # generate obmc input
        for batch_label in range(self.batch_size):
            
            # get position
            position = [pos_height[batch_label,0], pos_width[batch_label,0]]
            position_batch.append([int(position[0]*self.downsample), int(position[1]*self.downsample)])
            
            # get motion
            motion = [action_mot[batch_label,0], action_mot[batch_label,1]]
            motion_batch.append(motion)
            
            # update mask
            down_mask_size = int(self.mask_size/self.downsample)
            for i in range(-1 * down_mask_size, down_mask_size + 1, 1):
                for j in range(-1 * down_mask_size, down_mask_size + 1, 1):

                    # position exist?
                    if self.wrong_num(int(position[0]) + i, self.down_height) or self.wrong_num(int(position[1]) + j, self.down_width):
                        continue

                    self.mask[batch_label][int(position[0]) + i, int(position[1]) + j, 0] = 1
            
            # update position map
            position_mask = 2
            position = [int(position[0]*self.downsample), int(position[1]*self.downsample)]
            for i in range(-1 * position_mask, position_mask + 1, 1):
                for j in range(-1 * position_mask, position_mask + 1, 1):

                    # position exist?
                    if self.wrong_num(position[0] + i, self.height) or self.wrong_num(position[1] + j, self.width):
                        continue

                    self.position_map[batch_label][position[0] + i, position[1] + j, 0] = motion[0]
                    self.position_map[batch_label][position[0] + i, position[1] + j, 1] = motion[1]
        
        return position_batch, motion_batch, self.base_frame, self.base_frame_rgb
    
    
    def state_update(self, mc_frame, mc_frame_rgb):
        self.mc_seq.append(mc_frame)
        self.mc_seq_rgb.append(mc_frame_rgb)
    
    
    def load_frame(self, name, shift):
  
        # route
        route = ROOT_TO_CIF_DIR + '%s/%d.png'%(self.dataset_name, name+shift)
        return imread(route)

    
    def load_frame_rgb(self, name, shift):
  
        # route
        route = ROOT_TO_CIF_DIR + '%s/%d.png'%(self.dataset_name, name+shift)
        return imread(route)
    
    
    def load_image_rgb(self, current_label, shift, continue_time):
        
        if continue_time == 0:
            path = "./temp"
            if not os.path.isdir(path):
                os.mkdir(path)
            img = self.load_frame_rgb(current_label, shift)
            imsave("./temp/frame_rgb_%d.png"%(current_label + shift), img)
            return img / 255.0
        else:
            route = './temp/frame_rgb_%d.png'%(current_label + shift)
            img = imread(route)
            return img / 255.0
    
    
    def load_image(self, current_label, shift, continue_time):
        
        if continue_time == 0:
            path = "./temp"
            if not os.path.isdir(path):
                os.mkdir(path)
            img = self.load_frame(current_label, shift)
            imsave("./temp/frame_%d.png"%(current_label + shift), img)
            return np.expand_dims(img / 255.0, 2)
        else:
            route = './temp/frame_%d.png'%(current_label + shift)
            img = imread(route)
            return np.expand_dims(img / 255.0, 2)
    
    
    def save_image(self, image, image_rgb, position_map, current_label, continue_time):
        
        jump = 1
        image = np.reshape(image, (self.height, self.width))
        imsave("./temp/frame_%d.png"%(current_label + 3*jump + continue_time + 1), image)
        
        image_rgb = np.reshape(image_rgb, (self.height, self.width, 3))
        imsave("./temp/frame_rgb_%d.png"%(current_label + 3*jump + continue_time + 1), image_rgb)
        
        path = "./temp/position_map"
        if not os.path.isdir(path):
            os.mkdir(path)
        position_map = np.reshape(position_map, (self.height, self.width, 2))
        np.save("./temp/position_map/map_%d.npy"%(current_label), position_map)
    
    
    def wrong_num(self, num, max_value):
      
        # wrong number (not in 0~max_value)
        if num >= max_value or num < 0:
          return 1

        return 0
    
    
    def get_dataset_name(self):
        return self.dataset_name
    
    
    
    

