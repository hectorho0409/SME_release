import numpy as np
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt                     # plot
from matplotlib import pyplot as plt                # plot
from scipy.misc import imsave                       # imsave
import os                                           # build folder
import shutil                                       # delete old folder

class Visualization(object):
    def __init__(self, height, width, folder_name, task_name, total_frame):
    
        self.height = height
        self.width = width
        self.folder_name = folder_name
        self.task_name = task_name
        self.total_frame = total_frame
        self.initial()
        
        # delete old folder
        if os.path.isdir("./%s/%s_table"%(folder_name, task_name)):
            shutil.rmtree("./%s/%s_table"%(folder_name, task_name))
        
        # build new folder
        if not os.path.isdir("./%s"%folder_name):
            os.mkdir("./%s"%folder_name)
        if not os.path.isdir("./%s/%s_table"%(folder_name, task_name)):
            os.mkdir("./%s/%s_table"%(folder_name, task_name))
            os.mkdir("./%s/%s_table/current_frame"%(folder_name, task_name))
            os.mkdir("./%s/%s_table/predict_frame"%(folder_name, task_name))
            os.mkdir("./%s/%s_table/target_frame"%(folder_name, task_name))
            os.mkdir("./%s/%s_table/residual_frame"%(folder_name, task_name))
            os.mkdir("./%s/%s_table/total_frame"%(folder_name, task_name))
        
        # build text
        file = open('./%s/%s_table/ep_mean_mse.txt'%(folder_name, task_name), 'w')
        file.close()
        file = open('./%s/%s_table/ep_mean_psnr.txt'%(folder_name, task_name), 'w')
        file.close()
        file = open('./%s/%s_table/ep_mean_reward.txt'%(folder_name, task_name), 'w')
        file.close()
        file = open('./%s/%s_table/ep_mean_ssim.txt'%(folder_name, task_name), 'w')
        file.close()
        file = open('./%s/%s_table/ep_mean_info.txt'%(folder_name, task_name), 'w')
        file.close()
    
    def initial(self):
        
        self.epoch_num = 0
        self.mean_mse, self.mean_psnr, self.mean_reward, self.mean_ssim = [], [], [], []                    # mean result
        
        self.epoch_frame_label = 0
        self.copylast_mse, self.copylast_psnr, self.copylast_reward, self.copylast_ssim = [], [], [], []    # copylast result
        self.pretrain_mse, self.pretrain_psnr, self.pretrain_reward, self.pretrain_ssim = [], [], [], []    # pretrain result
        self.predict_mse, self.predict_psnr, self.predict_reward, self.predict_ssim = [], [], [], []        # predict result
        
        self.reset_epoch_info()
    
    def reset_epoch_info(self):
        self.eva_mse, self.eva_psnr, self.eva_reward, self.eva_ssim = 0., 0., 0., 0.                        # testing info
    
    def update_mean_info(self, mse, psnr, reward, ssim):
        
        # save new experiment result
        self.epoch_num += 1
        self.mean_mse.append(mse)
        self.mean_psnr.append(psnr)
        self.mean_reward.append(reward)
        self.mean_ssim.append(ssim)
        
        # update table
        self.plot_mean_mse_table()
        self.plot_mean_psnr_table()
        self.plot_mean_reward_table()
        self.plot_mean_ssim_table()
        
        # update text
        file = open('./%s/%s_table/ep_mean_mse.txt'%(self.folder_name, self.task_name), 'a')
        file.write('epoch: %i, %s-mse: %f\n' % (self.epoch_num-1, self.task_name, mse))
        file.close()
        file = open('./%s/%s_table/ep_mean_psnr.txt'%(self.folder_name, self.task_name), 'a')
        file.write('epoch: %i, %s-psnr: %f\n' % (self.epoch_num-1, self.task_name, psnr))
        file.close()
        file = open('./%s/%s_table/ep_mean_reward.txt'%(self.folder_name, self.task_name), 'a')
        text_pos = 'epoch: %i, %s-reward: %f\n' % (self.epoch_num-1, self.task_name, reward)
        file.write(text_pos)
        file.close()
        print(text_pos)
        file = open('./%s/%s_table/ep_mean_ssim.txt'%(self.folder_name, self.task_name), 'a')
        file.write('epoch: %i, %s-ssim: %f\n' % (self.epoch_num-1, self.task_name, ssim))
        file.close()
        file = open('./%s/%s_table/ep_mean_info.txt'%(self.folder_name, self.task_name), 'a')
        file.write('epoch: %i, %f %f %f\n' % (self.epoch_num-1, mse, psnr, ssim))
        file.close()
    
    def update_copylast_info(self, mse, psnr, reward, ssim):
        
        # save copy info
        self.copylast_mse.append(mse)
        self.copylast_psnr.append(psnr)
        self.copylast_reward.append(reward)
        self.copylast_ssim.append(ssim)
    
    def update_pretrain_and_predict_info(self, mse, psnr, reward, ssim):
        
        # update experiment process (frame level)
        self.epoch_frame_label += 1
        
        # save frame pretrain info (first training result)
        if self.epoch_num == 0:
            self.pretrain_mse.append(mse)
            self.pretrain_psnr.append(psnr)
            self.pretrain_reward.append(reward)
            self.pretrain_ssim.append(ssim)
        
        # save frame training info
        self.predict_mse.append(mse)
        self.predict_psnr.append(psnr)
        self.predict_reward.append(reward)
        self.predict_ssim.append(ssim)
        
        # save epoch training info
        self.eva_mse += mse
        self.eva_psnr += psnr
        self.eva_reward += reward
        self.eva_ssim += ssim
        
        # update table and reset next experiment
        if self.epoch_frame_label >= self.total_frame:
            
            # update frame training table
            self.plot_frame_mse_table()
            self.plot_frame_psnr_table()
            self.plot_frame_reward_table()
            self.plot_frame_ssim_table()
            
            # update epoch training table
            self.eva_mse = self.eva_mse/self.total_frame
            self.eva_psnr = self.eva_psnr/self.total_frame
            self.eva_reward = self.eva_reward/self.total_frame
            self.eva_ssim = self.eva_ssim/self.total_frame
            self.update_mean_info(self.eva_mse,self.eva_psnr,self.eva_reward,self.eva_ssim)
            
            # reset next experiment
            self.epoch_frame_label = 0
            self.predict_mse, self.predict_psnr, self.predict_reward, self.predict_ssim = [], [], [], []
            self.reset_epoch_info()
    
    def save_image(self, predict, current, target):
        
        # reshape
        current = np.reshape(current, (self.height, self.width))
        predict = np.reshape(predict, (self.height, self.width))
        target = np.reshape(target, (self.height, self.width))
        residual = np.abs(current - target)
        
        # save image
        #imsave('./%s/%s_table/current_frame/current_%d.png'%(self.folder_name, self.task_name, self.epoch_frame_label), current)
        #imsave('./%s/%s_table/predict_frame/predict_%d.png'%(self.folder_name, self.task_name, self.epoch_frame_label), predict)
        #imsave('./%s/%s_table/target_frame/target_%d.png'%(self.folder_name, self.task_name, self.epoch_frame_label), target)
        #imsave('./%s/%s_table/residual_frame/residual_%d.png'%(self.folder_name, self.task_name, self.epoch_frame_label), residual)
        
        imsave('./%s/%s_table/total_frame/frame_%d_current.png'%(self.folder_name, self.task_name, self.epoch_frame_label), current)
        imsave('./%s/%s_table/total_frame/frame_%d_predict.png'%(self.folder_name, self.task_name, self.epoch_frame_label), predict)
        imsave('./%s/%s_table/total_frame/frame_%d_target.png'%(self.folder_name, self.task_name, self.epoch_frame_label), target)
    
    def plot_mean_mse_table(self):
        
        # reshape
        y = np.reshape(self.mean_mse, (self.epoch_num))
        
        # design table
        a, = plt.plot(range(self.epoch_num), y, 'g', label='Line 1')
        b, = plt.plot(range(self.epoch_num), y, 'r.')
        
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        title_name = "RL - MSE Table"
        plt.title(title_name)
        save_route = "./%s/%s_table/mean_mse_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()

    def plot_mean_psnr_table(self):
    
        # reshape
        y = np.reshape(self.mean_psnr, (self.epoch_num))
        
        # design table
        a, = plt.plot(range(self.epoch_num), y, 'g', label='Line 1')
        b, = plt.plot(range(self.epoch_num), y, 'r.')
        
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        title_name = "RL - PSNR Table"
        plt.title(title_name)
        save_route = "./%s/%s_table/mean_psnr_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
    
    def plot_mean_reward_table(self):
        
        # reshape
        y = np.reshape(self.mean_reward, (self.epoch_num))
        
        # design table
        a, = plt.plot(range(self.epoch_num), y, 'g', label='Line 1')
        b, = plt.plot(range(self.epoch_num), y, 'r.')
        
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        title_name = "RL - Reward Table"
        plt.title(title_name)
        save_route = "./%s/%s_table/mean_reward_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
    
    def plot_mean_ssim_table(self):
        
        # reshape
        y = np.reshape(self.mean_ssim, (self.epoch_num))
        
        # design table
        a, = plt.plot(range(self.epoch_num), y, 'g', label='Line 1')
        b, = plt.plot(range(self.epoch_num), y, 'r.')
        
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        title_name = "RL - SSIM Table"
        plt.title(title_name)
        save_route = "./%s/%s_table/mean_ssim_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
    
    def plot_frame_mse_table(self):
        
        # reshape
        y_copylast = np.reshape(self.copylast_mse, (self.total_frame))
        y_pretrain = np.reshape(self.pretrain_mse, (self.total_frame))
        y_predict = np.reshape(self.predict_mse, (self.total_frame))
        
        # design loss table
        a, = plt.plot(range(self.total_frame), y_copylast, 'g', label='Line 1')
        #b, = plt.plot(range(self.total_frame), y_copylast, 'r.')
        c, = plt.plot(range(self.total_frame), y_pretrain,  'b', label='Line 2')
        #d, = plt.plot(range(self.total_frame), y_pretrain,  'r.')
        e, = plt.plot(range(self.total_frame), y_predict,  'y', label='Line 3')
        #f, = plt.plot(range(self.total_frame), y_predict,  'r.')
        
        plt.xlabel('frame label')
        plt.ylabel('MSE')
        title_name = "RL - MSE Table"
        plt.title(title_name)
        plt.legend((a, c, e), ("CopyLast", "Pretrain", "RL"))
        save_route = "./%s/%s_table/frame_mse_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
        
        # initial text
        file = open('./%s/%s_table/frame_mse.txt'%(self.folder_name, self.task_name), 'w')
        file.close()
        
        # save number in text
        file = open('./%s/%s_table/frame_mse.txt'%(self.folder_name, self.task_name), 'a')
        for i in range(self.total_frame):
            text_pos = 'Frame: %2d, Copylast: %6.4f, Pretrain: %6.4f, Predict: %6.4f\n' % (i, y_copylast[i], y_pretrain[i], y_predict[i])
            file.write(text_pos)
        file.close()
    
    def plot_frame_psnr_table(self):
        
        # reshape
        y_copylast = np.reshape(self.copylast_psnr, (self.total_frame))
        y_pretrain = np.reshape(self.pretrain_psnr, (self.total_frame))
        y_predict = np.reshape(self.predict_psnr, (self.total_frame))
        
        # design loss table
        a, = plt.plot(range(self.total_frame), y_copylast, 'g', label='Line 1')
        #b, = plt.plot(range(self.total_frame), y_copylast, 'r.')
        c, = plt.plot(range(self.total_frame), y_pretrain,  'b', label='Line 2')
        #d, = plt.plot(range(self.total_frame), y_pretrain,  'r.')
        e, = plt.plot(range(self.total_frame), y_predict,  'y', label='Line 3')
        #f, = plt.plot(range(self.total_frame), y_predict,  'r.')
        
        plt.xlabel('frame label')
        plt.ylabel('PSNR')
        title_name = "RL - PSNR Table"
        plt.title(title_name)
        plt.legend((a, c, e), ("CopyLast", "Pretrain", "RL"))
        save_route = "./%s/%s_table/frame_psnr_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
        
        # initial text
        file = open('./%s/%s_table/frame_psnr.txt'%(self.folder_name, self.task_name), 'w')
        file.close()
        
        # save number in text
        file = open('./%s/%s_table/frame_psnr.txt'%(self.folder_name, self.task_name), 'a')
        for i in range(self.total_frame):
            text_pos = 'Frame: %2d, Copylast: %6.4f, Pretrain: %6.4f, Predict: %6.4f\n' % (i, y_copylast[i], y_pretrain[i], y_predict[i])
            file.write(text_pos)
        file.close()
    
    def plot_frame_reward_table(self):
    
        # reshape
        y_copylast = np.reshape(self.copylast_reward, (self.total_frame))
        y_pretrain = np.reshape(self.pretrain_reward, (self.total_frame))
        y_predict = np.reshape(self.predict_reward, (self.total_frame))
        
        # design loss table
        a, = plt.plot(range(self.total_frame), y_copylast, 'g', label='Line 1')
        #b, = plt.plot(range(self.total_frame), y_copylast, 'r.')
        c, = plt.plot(range(self.total_frame), y_pretrain,  'b', label='Line 2')
        #d, = plt.plot(range(self.total_frame), y_pretrain,  'r.')
        e, = plt.plot(range(self.total_frame), y_predict,  'y', label='Line 3')
        #f, = plt.plot(range(self.total_frame), y_predict,  'r.')
        
        plt.xlabel('frame label')
        plt.ylabel('Reward')
        title_name = "RL - Reward Table"
        plt.title(title_name)
        plt.legend((a, c, e), ("CopyLast", "Pretrain", "RL"))
        save_route = "./%s/%s_table/frame_reward_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
        
        # initial text
        file = open('./%s/%s_table/frame_reward.txt'%(self.folder_name, self.task_name), 'w')
        file.close()
        
        # save number in text
        file = open('./%s/%s_table/frame_reward.txt'%(self.folder_name, self.task_name), 'a')
        for i in range(self.total_frame):
            text_pos = 'Frame: %2d, Copylast: %6.4f, Pretrain: %6.4f, Predict: %6.4f\n' % (i, y_copylast[i], y_pretrain[i], y_predict[i])
            file.write(text_pos)
        file.close()
    
    def plot_frame_ssim_table(self):
    
        # reshape
        y_copylast = np.reshape(self.copylast_ssim, (self.total_frame))
        y_pretrain = np.reshape(self.pretrain_ssim, (self.total_frame))
        y_predict = np.reshape(self.predict_ssim, (self.total_frame))
        
        # design loss table
        a, = plt.plot(range(self.total_frame), y_copylast, 'g', label='Line 1')
        #b, = plt.plot(range(self.total_frame), y_copylast, 'r.')
        c, = plt.plot(range(self.total_frame), y_pretrain,  'b', label='Line 2')
        #d, = plt.plot(range(self.total_frame), y_pretrain,  'r.')
        e, = plt.plot(range(self.total_frame), y_predict,  'y', label='Line 3')
        #f, = plt.plot(range(self.total_frame), y_predict,  'r.')
        
        plt.xlabel('frame label')
        plt.ylabel('SSIM')
        title_name = "RL - SSIM Table"
        plt.title(title_name)
        plt.legend((a, c, e), ("CopyLast", "Pretrain", "RL"))
        save_route = "./%s/%s_table/frame_ssim_table.png"%(self.folder_name, self.task_name)
        plt.savefig(save_route)
        plt.close()
        
        # initial text
        file = open('./%s/%s_table/frame_ssim.txt'%(self.folder_name, self.task_name), 'w')
        file.close()
        
        # save number in text
        file = open('./%s/%s_table/frame_ssim.txt'%(self.folder_name, self.task_name), 'a')
        for i in range(self.total_frame):
            text_pos = 'Frame: %2d, Copylast: %6.4f, Pretrain: %6.4f, Predict: %6.4f\n' % (i, y_copylast[i], y_pretrain[i], y_predict[i])
            file.write(text_pos)
        file.close()
    
    def wrong_num(num, max_value):
        if num >= max_value or num < 0:
            return 1
        return 0

    def plot_image(self, imgc, pos_seq, frame_label):
        
        # get kernel
        mask_size = 2
        
        # position number
        seq_num = (np.shape(pos_seq))[0]
        
        # get rgb image
        z = np.zeros((self.hight, self.width, 3))
        z[:,:,0] = imgc[0, :, :, 0]
        z[:,:,1] = imgc[0, :, :, 0]
        z[:,:,2] = imgc[0, :, :, 0]
        
        for k in range(seq_num):
            for i in range(-1*mask_size, mask_size+1):
                for j in range(-1*mask_size, mask_size+1):
                    
                    # position [pos[0]+i, pos[1]+j] exist?
                    if self.wrong_num(pos_seq[k][0]+i, self.hight) or self.wrong_num(pos_seq[k][1]+j, self.width):
                        continue

                    # plot position dot
                    z[pos_seq[k][0]+i, pos_seq[k][1]+j, 0] = 0. # R
                    z[pos_seq[k][0]+i, pos_seq[k][1]+j, 1] = 1. # G
                    z[pos_seq[k][0]+i, pos_seq[k][1]+j, 2] = 0. # B
        
        # save
        filename = "./%s/frame_%d.png"%(self.folder_name, frame_label)
        imsave(filename, z)
    
