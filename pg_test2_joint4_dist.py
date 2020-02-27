import numpy as np
import tensorflow as tf

from load_data7 import Load_cif_test as Load_TestData
from environment5 import Env_toolbox as Env_gray
from environment5_rgb import Env_toolbox as Env_rgb
from model_dist4_test import PolicyEstimator as Model

# dataset parameter
HEIGHT, WIDTH = 288, 352
MAX_STEPS = 80
TEST_FRAME_PAR = 290

# model parameter
DOWNSAMPLE = 4


def reinforce(model, save_folder):

    # load dataset route
    first_seq_label = 0
    
    # initial testing tool
    dataset_package_test = Load_TestData(HEIGHT, WIDTH, DOWNSAMPLE, 1, [])
    
    env_gray, env_rgb, continue_total = [], [], 1
    for continue_time in range(continue_total):
        # gray part
        env_gray.append(Env_gray(MAX_STEPS, 1, HEIGHT, WIDTH))
        # rgb part
        one_rgb = Env_rgb(MAX_STEPS, 1, HEIGHT, WIDTH)
        one_rgb.build_visualization(save_folder + "%d"%continue_time, TEST_FRAME_PAR)
        env_rgb.append(one_rgb)
    
    # testing
    first_test_label = first_seq_label + 0
    for group_label in range(first_test_label, first_test_label + TEST_FRAME_PAR, 1):
        
        print("  Test-Process: %s sequences, %d steps, %d times, %5d/%5d th dataset label"%(dataset_package_test.get_dataset_name(), MAX_STEPS, continue_total, group_label-first_test_label, TEST_FRAME_PAR-1))
        
        for continue_time in range(continue_total):
            
            # test one game
            for step_label in range(MAX_STEPS):
                # predict action1 (position part)
                state, state_pad, state_mask, _ = dataset_package_test.get_model_input1(group_label, continue_time)
                _, pos_height, pos_width = model.play_position(state, state_mask, False, 1)
                # predict action2 (motion part)
                down_pos, _ = dataset_package_test.get_model_input2(pos_height, pos_width)
                action_mot = model.play_motion(state_pad, down_pos, False, 1)
                # predict action3 (region adaptive part)
                _, order = model.play_region_adaptive(down_pos, action_mot, state_pad, False, 1)
                # environment update
                real_pos, motion, base_frame, base_frame_rgb = dataset_package_test.get_env_input(pos_height, pos_width, action_mot)
                mc_frame = env_gray[continue_time].do_obmc(real_pos, motion, np.ones((1, 1))*0.5, base_frame)         # order, np.ones((1, 1))*1.5
                mc_frame_rgb = env_rgb[continue_time].do_obmc(real_pos, motion, np.ones((1, 1))*0.5, base_frame_rgb)  # order, np.ones((1, 1))*1.5
                # state update
                dataset_package_test.state_update(mc_frame, mc_frame_rgb)
            
            # reset your game
            reference_frame, object_frame, mc_seq, mc_seq_rgb, position_map = dataset_package_test.game_over()
            env_gray[continue_time].game_over()
            env_rgb[continue_time].game_over()
            
            # update reward and save test info
            env_rgb[continue_time].update_test_info(reference_frame, reference_frame, object_frame, "Save_Copy")
            env_rgb[continue_time].update_test_info(mc_seq_rgb[MAX_STEPS-1], reference_frame, object_frame, "Save_Test")
            
            # save predict frame as next frame
            dataset_package_test.save_image(mc_seq[MAX_STEPS-1], mc_seq_rgb[MAX_STEPS-1], position_map, group_label, continue_time)
        

def main(_):
    
    # I/O folder
    read_folder = "RL_pg_training_task0"  # RL_pg_training_task0, RL_pg_pretrain_policy_20
    save_folder = "RL_pg_test_"
    
    # build model
    model = Model(HEIGHT, WIDTH, DOWNSAMPLE)
    
    # Restore weight
    model.restore_weight(read_folder, "region_adaptive")
    model.restore_weight(read_folder, "position")
    model.restore_weight(read_folder, "motion")
    print("")
    
    # RL training
    reinforce(model, save_folder)

if __name__ == '__main__':
    tf.app.run()
