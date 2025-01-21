"""
This python file does:

- defines a function 
  - "collect_data_simulation" to controll the robot and collect training data
  - "save_data_to_zarr" saves collected data to a zarr storage
    

"""
# IMPORTS

import numpy as np
import pybullet as p
import cv2
from robo_env import PushCubeEnv # load robot controll environment
import time
import zarr
import os
import datetime


def collect_data_simulation(
        generateVideo: bool = False, # video: should a preview video be generated?
        saveTrainingData: bool =  False,
        saveTrainingPath = "data_storage.zarr",
        output_filename: str = 'output_video.avi',
        fps: float = 20, # Frames per second - sim speed
        useMouse : bool= False, # use mouse for demonstration? if yes, left click and hold = move EE in x and y. arrow up = + z, arrow down = - z. controll gripper with slider
        img_width : int = 96, # image width that is recorded to images data for training and also for the video
        img_height : int = 96,
    ):
    """ run sim and collect training data and save it as zarr storage for training diffusion model """
    sim = PushCubeEnv(width=img_width, height=img_height, useMouse=useMouse) # set width and height
    
  
    
    # Initialize OpenCV VideoWriter
    if generateVideo: # only record if the video should be genrated
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' or 'MJPG'
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (sim.width, sim.height))
    # Data storage
    images, segmentedImages, states, actions, episode_ends = [], [], [], [], []
    reset_counter = 1
    step = 1 # initialize step

    if saveTrainingData:

        # Create a root group for the Zarr store
        
        root = zarr.open(saveTrainingPath, mode='w')

        if os.path.exists(os.path.join(os.getcwd(), saveTrainingPath)):
            print("data_storage.zarr already exists! Will create a new folder")
            x = datetime.datetime.now().strftime("%d_%m_%y")
            root = zarr.open(f"data_storage_{x}.zarr", mode='w')
        

        # Create Zarr group for 'data' 
        data_group = root.create_group('data')
        '''
        data_storage.zarr
        II--> data
        '''

        # Create subgroup 'meta'
        meta_group = root.create_group('meta')
        
        '''
        data_storage.zarr
        II--> data
        II--> meta
        '''
    

    # Initialize variables to store previous state
    prev_state = None

    while True:
        sim.process_events()
        #p.stepSimulation()  # Make sure simulation steps are being processed
        
        
        #print(f"get_current_pose(self): {sim.read_gui_mouse()}")
        if useMouse:
            x, y, z, Rx, Ry, Rz, finger = sim.read_gui_mouse()
        else:
            # Read current slider values
            x, y, z, Rx, Ry, Rz, finger = sim.read_gui_sliders()
            # get joint angles from EE values
            joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
            # set joint values
            sim.set_joint_angles(joint_angles)

        current_state = np.array([x, y, z, Rx, Ry, Rz, finger])
        # Check if the state has changed
        state_changed = prev_state is None or not np.allclose(current_state, prev_state, atol=1e-6 if not useMouse else 1e-3)

        if state_changed:
            

            # Append data only if state has changed
            actions.append(current_state.tolist())
            images.append(sim.get_camera_image())
            segmentedImages.append(sim.get_camera_image(segmentation=True))
            states.append(current_state.tolist())

            # Update previous state
            prev_state = current_state

            if generateVideo:
                video_writer.write(sim.get_camera_image(segmentation=True))  # Write frame to video

            time.sleep(1/fps)

            step += 1  # increase step now
            print(f"Step: {step}. Press r to reset and x to close. Resets: {reset_counter}")

        # Check for keyboard events
        keys = p.getKeyboardEvents()        

        # Check if the 'r' key has been pressed to reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("Resetting...")
            episode_ends.append(step)  # log the end of the episode as step, step counts throughout all episodes     

            reset_counter += 1  # increase reset_counter            
            prev_state = None  # Reset previous state
            sim.reset()  # reset env

        

        # Check if the 'x' key has been pressed to exit
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
            print("Exiting loop...")

            # Save the collected data to Zarr
            if saveTrainingData:
                save_data_to_zarr(data_group, images, segmentedImages, actions, states)

                # Save episode ends
                meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int32))
                episode_ends.append(step)
                print("Episode Ends: ", episode_ends)
               
            break
        
    # Release the video writer
    if generateVideo:
        video_writer.release()
        cv2.destroyAllWindows()


def collect_data_simulation_angles(
        generateVideo: bool = False, # video: should a preview video be generated?
        saveTrainingData: bool =  False,
        saveTrainingPath = "angles_data_storage.zarr",
        output_filename: str = 'output_video.avi',
        demonstrationsCount = 200, # how many epochs should be recorded?
        timeCountLimit = 5, # how many seconds for the task to be completed?
        distanceThresholdCubetoGoal = 0.02, # how close should the cube be to the target in order to count as done?
        fps: float = 20, # Frames per second - sim speed
        useMouse : bool= False, # use mouse for demonstration? if yes, left click and hold = move EE in x and y. arrow up = + z, arrow down = - z. controll gripper with slider
        img_width : int = 96, # image width that is recorded to images data for training and also for the video
        img_height : int = 96,
    ):
    """ program showing how to use the sim to collect training data but with ANGLES and save it as zarr storage for training diffusion model """
    sim = PushCubeEnv(width=img_width, height=img_height, useMouse=useMouse) # set width and height
    
  
    
    # Initialize OpenCV VideoWriter
    if generateVideo: # only record if the video should be genrated
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' or 'MJPG'
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (sim.width, sim.height))
    # Data storage
    images, segmentedImages, states, actions, episode_ends = [], [], [], [], []
    reset_counter = 1
    step = 1 # initialize step

    if saveTrainingData:

        # Create a root group for the Zarr store      

        if os.path.exists(os.path.join(os.getcwd(), saveTrainingPath)):
            print(f"{saveTrainingPath} already exists! Will create a new folder")
            x = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
            root = zarr.open(f"angles_data_storage_{x}.zarr", mode='w')
        
        else:
            print(f"Created {saveTrainingPath} folder.")
            root = zarr.open(saveTrainingPath, mode='w')
        # Create Zarr group for 'data' 
        data_group = root.create_group('data')
        '''
        data_storage.zarr
        II--> data
        '''

        # Create subgroup 'meta'
        meta_group = root.create_group('meta')
        
        '''
        data_storage.zarr
        II--> data
        II--> meta
        '''
    

    # Initialize variables to store previous state
    #prev_state = None
    timeCount  = 0
    # Get the starting time
    # start_time = time.time()
    while True:
        sim.process_events()
       
        #print(f"Joint angles: {sim.get_joint_angles()}")
        current_state = np.array(sim.get_joint_angles())
        # Check if the state has changed
        #state_changed = prev_state is None or not np.allclose(current_state, prev_state, atol=1e-6 if not useMouse else 1e-3)

        #if state_changed:            

        # Append data only if state has changed
        actions.append(current_state.tolist())
        images.append(sim.get_camera_image())
        segmentedImages.append(sim.get_camera_image(segmentation=True))
        states.append(current_state.tolist())

        # Update previous state
        #prev_state = current_state

        if generateVideo:
            video_writer.write(sim.get_camera_image(segmentation=True))  # Write frame to video

        time.sleep(1/fps)
        # Update timeCount in real-time
        timeCount += 1/fps#time.time() - start_time
        step += 1  # increase step now
        print(f"TimeCount: {timeCount:.2f}, Step: {step}. Press r to reset and x to close. Resets: {reset_counter}")

        # Check for keyboard events
        keys = p.getKeyboardEvents()        

        # Check if the 'r' key has been pressed to reset
        if (ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED) or (timeCount >= timeCountLimit) or sim.isDone(distanceThresholdCubetoGoal):
            print("Resetting...")
            episode_ends.append(step)  # log the end of the episode as step, step counts throughout all episodes     

            reset_counter += 1  # increase reset_counter 
            # Reset timeCount and start_time
            timeCount = 0
            # start_time = time.time()           
            #prev_state = None  # Reset previous state
            sim.reset()  # reset env
        

        # Check if the 'x' key has been pressed to exit
        if (ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED) or reset_counter >= demonstrationsCount:
            print("Exiting loop...")

            # Save the collected data to Zarr
            if saveTrainingData:
                save_data_to_zarr(data_group, images, segmentedImages, actions, states)

                # Save episode ends
                meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int32))
                episode_ends.append(step)
                print("Episode Ends: ", episode_ends)
               
            break
        
    # Release the video writer
    if generateVideo:
        video_writer.release()
        cv2.destroyAllWindows()   

def save_data_to_zarr(data_group, images, segmentedImages, actions, states):
    # Save images
    img_shape = (len(images), *images[0].shape) # setup image zarr
    img_chunks = (1, *images[0].shape)  # One image sequence per chunk
    img_dataset = data_group.create_dataset('img', 
                                            shape=img_shape, 
                                            chunks=img_chunks, 
                                            dtype=np.uint8)
    #for i, img in enumerate(images):
    img_dataset[:] = np.array(images, dtype=np.uint8)

    # Save segmented images
    seg_img_shape = (len(segmentedImages), *segmentedImages[0].shape) # setup image zarr
    seg_img_chunks = (1, *segmentedImages[0].shape)  # One image sequence per chunk
    seg_img_dataset = data_group.create_dataset('seg_img', 
                                            shape=seg_img_shape, 
                                            chunks=seg_img_chunks, 
                                            dtype=np.uint8)
    #for i, img in enumerate(segmentedImages):
    seg_img_dataset[:] = np.array(segmentedImages, dtype=np.uint8)

    # Save actions
    action_shape = (len(actions), len(actions[0]))
    action_dataset = data_group.create_dataset('action', 
                                               shape=action_shape, 
                                               chunks=(1, len(actions[0])), 
                                               dtype=np.float32)
    
    action_dataset[:] = np.array(actions)
    #print("actions: ", np.array(actions))

    # Save states
    state_shape = (len(states), len(states[0]))
    state_dataset = data_group.create_dataset('state', 
                                              shape=state_shape, 
                                              chunks=(1, len(states[0])), 
                                              dtype=np.float32)
    state_dataset[:] = np.array(states)
    #print("states: ", np.array(states))

if __name__ == "__main__":
    collect_data_simulation_angles(
        generateVideo = False, # video: should a preview video be generated?
        saveTrainingData =  True, # should data be saved?
        saveTrainingPath = "angles_data_storage.zarr",
        demonstrationsCount = 400,        
        timeCountLimit = 5, # how many in-simulation seconds for the task to be completed?
        distanceThresholdCubetoGoal = 0.02, # how close should the cube be to the target in order to count as done?
        output_filename = 'output_video.avi',
        fps = 20, # Frames per second - sim speed
        useMouse = True, # use mouse for demonstration? if yes, left click and hold = move EE in x and y. arrow up = + z, arrow down = - z. controll gripper with slider
        img_width = 96, # image width that is recorded to images data for training and also for the video
        img_height = 96,
    )


