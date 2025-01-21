"""
This python file does:
- defines the function run_simulation() to iteretavily run the environment and benchmark a model on that task
- it defines two inference functions, which workd for seperate model versions:
infer_action_sample - works for trained diffusion models, which predict samples directly
infer_action        - works for trained diffusion models, which predict noise residuals directly

"""

# IMPORTS
import numpy as np
import torch
import os
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import time
import cv2
import pybullet as p

# load robot controll environment
from robo_env import PushCubeEnv 
# import dataset functions
from dataset import *
# import u net functions
from network import *
# import vision encoder functions
from vision_encoder import *
# import utils for loading models, checking file paths, et.c
from utils import *


#@markdown ### **Inference**

#@markdown ### **Inference**
'''
    def infer_action(
            nimages, 
            nagent_poses, 
            stats, 
            num_diffusion_iters,
            ema_nets=nets,
            visualizeSampling=visualizeSampling
        ):

        # Measure time
        start_time = time.time()       
        # model to eval mode
        ema_nets.eval()
        #print(f"noise_scheduler.config: {noise_scheduler}")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=DDPMScheduler_training_steps,#6,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            # Get image features
            image_features = ema_nets['vision_encoder'](nimages.to(device))
            
            # Concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses.to(device)], dim=-1)
            
            # Reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            
            # initialize action from Guassian noise
            noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)
            
            # Init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters) #num_diffusion_iters            
            

            for timestep in noise_scheduler.timesteps:
                #print(f"timestep: {timestep}")
                noise_pred = ema_nets['noise_pred_net'](
                    sample=  noisy_action,
                    timestep= timestep,#*2,
                    global_cond=obs_cond
                ) 
                        
                noisy_action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=noisy_action
                ).prev_sample       
                                
                # Unnormalize intermediate actions
                if visualizeSampling == True:
                    action_pred = unnormalize_data(noisy_action.squeeze(0).cpu().numpy(), stats=stats['action'])[1:9,:3]
                    # print(f"action_pred: {action_pred}")
                    # visualize intermediate actions as spheres                
                    sim.update_path_spheres(action_pred)                 
                
        # Unnormalize action
        noisy_action = noisy_action.squeeze(0).cpu().numpy()
        action_pred = unnormalize_data(noisy_action, stats=stats['action'])
        # save time
        inferenceTime = (time.time() - start_time)
        #print(f"inferenceTime: {inferenceTime:.3f} s") 
        return action_pred, inferenceTime

    
    def infer_action_sample(
            nimages, 
            nagent_poses, 
            stats, 
            num_diffusion_iters,
            ema_nets=nets,
            visualizeSampling=visualizeSampling
        ):

        # Measure time
        start_time = time.time()       
        # model to eval mode
        ema_nets.eval()
        #print(f"noise_scheduler.config: {noise_scheduler}")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=DDPMScheduler_training_steps,#6,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='sample'
        )
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            # Get image features
            image_features = ema_nets['vision_encoder'](nimages.to(device))
            
            # Concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses.to(device)], dim=-1)
            
            # Reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            
            # initialize action from Guassian noise
            noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)
            
            # Init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters) #num_diffusion_iters            
            

            for timestep in noise_scheduler.timesteps:
                #print(f"timestep: {timestep}")
                sample_pred = ema_nets['sample_pred_net'](
                    noisy_sample=  noisy_action,
                    timestep= timestep,#*2,
                    global_cond=obs_cond
                ) 
                        
                noisy_action = noise_scheduler.step(
                    model_output=sample_pred,
                    timestep=timestep,
                    sample=noisy_action
                ).prev_sample       
                                
                # Unnormalize intermediate actions
                if visualizeSampling == True:
                    action_pred = unnormalize_data(noisy_action.squeeze(0).cpu().numpy(), stats=stats['action'])[1:9,:3]
                    # print(f"action_pred: {action_pred}")
                    # visualize intermediate actions as spheres                
                    sim.update_path_spheres(action_pred)                 
                
        # Unnormalize action
        noisy_action = noisy_action.squeeze(0).cpu().numpy()
        action_pred = unnormalize_data(noisy_action, stats=stats['action'])
        # save time
        inferenceTime = (time.time() - start_time)
        #print(f"inferenceTime: {inferenceTime:.3f} s") 
        return action_pred, inferenceTime
'''

def infer_action(
        nimages, 
        nagent_poses, 
        stats, 
        num_diffusion_iters,
        useNoisePred, # use noise prediction or sample?
        DDPMScheduler_training_steps, # ddpm scheduler steps!
        ema_nets,
        visualizeSampling,
        device,
        pred_horizon,
        action_dim,
        sim
    ):

    # Measure time
    start_time = time.time()       
    # model to eval mode
    ema_nets.eval()
    #print(f"noise_scheduler.config: {noise_scheduler}")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=DDPMScheduler_training_steps,#6,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type= 'epsilon' if useNoisePred else 'sample'
    )
    
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        # Get image features
        image_features = ema_nets['vision_encoder'](nimages.to(device))
        
        # Concat with low-dim observations
        obs_features = torch.cat([image_features, nagent_poses.to(device)], dim=-1)
        
        # Reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        
        # initialize action from Guassian noise
        noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)
        
        # Init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters) #num_diffusion_iters            
        

        for timestep in noise_scheduler.timesteps:
            #print(f"timestep: {timestep}")
            pred = ema_nets['noise_pred_net'](
                sample=  noisy_action,
                timestep= timestep,#*2,
                global_cond=obs_cond
            ) if useNoisePred else ema_nets['sample_pred_net'](
                noisy_sample=  noisy_action,
                timestep= timestep,#*2,
                global_cond=obs_cond
            ) 
                    
            noisy_action = noise_scheduler.step(
                model_output=pred,
                timestep=timestep,
                sample=noisy_action
            ).prev_sample       
                            
            # Unnormalize intermediate actions
            if visualizeSampling == True:
                action_pred = unnormalize_data(noisy_action.squeeze(0).cpu().numpy(), stats=stats['action'])[1:9,:3]
                # print(f"action_pred: {action_pred}")
                # visualize intermediate actions as spheres                
                sim.update_path_spheres(action_pred)                 
            
    # Unnormalize action
    noisy_action = noisy_action.squeeze(0).cpu().numpy()
    action_pred = unnormalize_data(noisy_action, stats=stats['action'])
    # save time
    inferenceTime = (time.time() - start_time)
    #print(f"inferenceTime: {inferenceTime:.3f} s") 
    return action_pred, inferenceTime
 
                                                    
def run_simulation(
        lowdim_obs_dim,
        action_dim,# dim of actions 
        dataset_path=os.path.join(os.getcwd(), "data_storage.zarr"), 
        model_path="", 
        useSegmented = False,  # use segmented image data
        inferenceEpochs = 100, 
        new_num_diffusion_iters = 10, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "data_inf", 
        visualizeSampling=False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),   
        vision_feature_dim = 512,
        obs_horizon = 2,
        pred_horizon = 16,
        action_horizon = 8,
        agent_pos_cutoff_position = 3,
        useNoisePred = True,        
        generateVideo = False, # video: should a preview video be generated?
        img_width = 96,
        img_height = 96,
        fps = 20,  # Frames per second bestimmt die Frequenz mit der die generierten Roboteraktionen ausgeführt werden
        distanceThresholdCubetoGoal = 0.02,
        timeOutSetps = 30 # wie viele Aktionssequenzen bis es als failed zählt?
        ):
    """
    model_path: relative path to the model 
    dataset_path: path to "data_storage.zarr" 
    useSegmented: should segmented images be used?
    inferenceEpochs: how many epochs should be tested?
    new_num_diffusion_iters: how many reverse diffusion steps?
    DDPMScheduler_training_steps: trainingsteps of the DDPM Scheduler
    inf_data_ouput_name:  output name of the inference data numpy array
    visualizeSampling:  visualize the sampling process
    device: device to run on
    """
    # check valid inf_data_ouput_name

    if is_valid_filename_practical(f"{inf_data_ouput_name}.npy") == False: print(f"Filename {inf_data_ouput_name} is not valid.")        
           
    # Check if dataset_path exists
    if os.path.exists(dataset_path):
        print(f"{dataset_path} exists.")
    else:
        print(f"{dataset_path} does not exist.")
        return

    # create dataset from file
    dataset = ImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        agent_pos_cutoff_position=agent_pos_cutoff_position,
        useSegmented=useSegmented
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # Load model nets

    nets = load_model_with_info(
        model_path, 
        device=device,
        lowdim_obs_dim=lowdim_obs_dim,
        vision_feature_dim=vision_feature_dim,
        obs_horizon=obs_horizon,
        action_dim=action_dim,
        useNoisePred=useNoisePred)

    if nets == 0:
        print("Model not loaded...quitting.")
        return
    
    # load robot sim
    sim = PushCubeEnv(width=img_width, height=img_height) # set width and height
    step = 1
    # Initialize video writer variables
    output_filename = 'output_video_seg.avi' if useSegmented else 'output_video.avi'
    

    # Initialize OpenCV VideoWriter
    if generateVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' or 'MJPG'
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (sim.width, sim.height))
    # Data storage
    images, states = [], []
    reset_counter = 0 # wie viele Resets?
    simIsDone = False
    infIsDone = False # is this inference loop done?

    done_counter = 0 # wie viele Erfolge?    
    inferenceTimes = []
    current_trajectory = [] # die aktuelle Lösungstrajektorie
    inferenceData = [] # die Daten der Lösungsschritte
    # Define the data type for the structured array
    dt = np.dtype([
        ('episode_num', np.int32),
        ('status', 'U10'),  # Unicode string up to 10 chars for "done"/"not done"
        ('success_rate', np.float32),
        ('inf_time_avg', np.float32),
        ('trajectory', object),  # For the trajectory list/array
        ('traj_length', np.int32),
        ('epoch_time', np.float32),
        ('cube_pos', np.float32, (3,))  # Assuming cube_pos is a 3D position
    ])    
    epochStartTime = 0
    epochTime = 0

    while infIsDone == False:
        pose, _ = sim.get_current_pose()
        x, y, z = pose[0], pose[1], pose[2]
        # stack the last obs_horizon number of observations
        if epochStartTime == 0: epochStartTime = time.time() # start time of epoch
        
        if len(images) <2: # fill first 2 poses and images
            images.append(sim.get_camera_image(segmentation=useSegmented))
            states.append([x,y,z])
            # record trajectory
            current_trajectory.append([x,y,z])
        else:
            obs_images = images[-2:]
            #images = np.stack([x['image'] for x in obs_deque]) # get 2 images
            obs_poses = states[-2:]
            # agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(obs_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            
            nimages = np.array(obs_images).astype(np.float32)  # Convert to float
            nimages /= 255.0  # Normalize to [0, 1]
            nimages = torch.tensor(nimages).permute(0, 3, 1, 2) 
            #print(f"nimages.shape: {nimages.shape}")

            # device transfer
            nimages = nimages.to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            action_pred, inferenceTime = 0,0
            # Inference             
            action_pred, inferenceTime = infer_action(
                nimages=nimages,
                nagent_poses=nagent_poses,
                stats=stats,
                num_diffusion_iters=new_num_diffusion_iters,
                ema_nets=nets, # give the nets!
                useNoisePred=useNoisePred, # use noise prediction or sample?
                DDPMScheduler_training_steps=DDPMScheduler_training_steps, # ddpm scheduler steps!                    
                device=device,
                pred_horizon=pred_horizon,
                action_dim=action_dim,
                sim=sim,
                visualizeSampling=visualizeSampling
            )
        
            
            inferenceTimes.append(inferenceTime)
            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            
            # execute action_horizon number of steps without replanning           
            for i in range(len(action)):
                # stepping env
                # append states and images again
                images.append(sim.get_camera_image(segmentation=useSegmented))
                states.append([action[i][0],action[i][1], action[i][2]])
                #print(f"action {i}: {[round(action[i][0], 2),round(action[i][1],2), round(action[i][2],2)]}") 
                current_trajectory.append([action[i][0],action[i][1], action[i][2]])
                
                joint_angles = sim.calculate_ik([action[i][0],action[i][1], action[i][2]], [action[i][3], action[i][4], action[i][5]])
                #print(f"joint_angles: {joint_angles}")
                sim.set_joint_angles(joint_angles)
                if generateVideo:
                    video_writer.write(sim.get_camera_image(segmentation=useSegmented))  # Write frame to video
                # check if done in every step
                if sim.isDone(distanceThresholdCubetoGoal): 
                    simIsDone = True
                    epochTime = time.time() - epochStartTime # time of epoch
                time.sleep(1/fps)

        # Check for keyboard events
        keys = p.getKeyboardEvents()
        # Check if the 'r' key has been pressed to reset                    # geschafft       # gefailt
        if (ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED) or simIsDone or step > timeOutSetps:
                      
            
            # handle success
            if simIsDone:
                done_counter+=1
            else:
                epochTime = time.time() - epochStartTime # time of epoch
            reset_counter+=1 # before tracking
            # record inference data
            inferenceData.append((reset_counter, #number of inference episode
                                "done" if simIsDone else "not done", #number of successfully episodes
                                done_counter/reset_counter if reset_counter != 0 else -1,  #success rate
                                sum(inferenceTimes) / len(inferenceTimes), #inf time average
                                current_trajectory, #current_trajectory
                                len(current_trajectory), # length of trajectory
                                epochTime, #time of epoch
                                sim.get_cube_pos())) # red cube pos
            
                        
            print(f"Episode: {reset_counter}, done: {simIsDone}, Success Rate: {done_counter/reset_counter if reset_counter != 0 else -1}, Average Inference Time: {sum(inferenceTimes) / len(inferenceTimes)}, Epoch Time: {epochTime}, Time left: {((sum(record[6] for record in inferenceData)/reset_counter)* (inferenceEpochs - reset_counter))/60} min.")   
                  
            # reset
            simIsDone = False
            images, states, current_trajectory= [], [], []
            
            step = 0
            epochTime = 0
            epochStartTime = 0
            sim.reset()
            
        
        step += 1
        # Check if the 'x' key has been pressed to exit
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED or reset_counter == inferenceEpochs:
            print("Exiting loop...") 
            # Convert to structured array
            inference_array = np.array(inferenceData, dtype=dt)  
            
            # save inference data                
            np.save(f'{inf_data_ouput_name}.npy', inference_array) 
            # Release the video writer
            if generateVideo:
                video_writer.release()
                cv2.destroyAllWindows()
            # break inference while loop
            infIsDone = True 
            sim.closeEnv()     
            break
    
def run_simulation_with_angles(
        lowdim_obs_dim,
        action_dim,# dim of actions         
        agent_pos_cutoff_position,
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage.zarr"), 
        model_path="", 
        useSegmented = False,  # use segmented image data
        inferenceEpochs = 100, 
        new_num_diffusion_iters = 10, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "data_inf", 
        visualizeSampling=False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),   
        vision_feature_dim = 512,
        obs_horizon = 2,
        pred_horizon = 16,
        action_horizon = 8,
        useNoisePred = True,        
        generateVideo = False, # video: should a preview video be generated?
        img_width = 96,
        img_height = 96,
        fps = 20,  # Frames per second bestimmt die Frequenz mit der die generierten Roboteraktionen ausgeführt werden
        distanceThresholdCubetoGoal = 0.02,
        timeOutSetps = 30, # wie viele Aktionssequenzen bis es als failed zählt?
        timeCountLimit = 5
        ):
    """
    model_path: relative path to the model 
    dataset_path: path to "data_storage.zarr" 
    useSegmented: should segmented images be used?
    inferenceEpochs: how many epochs should be tested?
    new_num_diffusion_iters: how many reverse diffusion steps?
    DDPMScheduler_training_steps: trainingsteps of the DDPM Scheduler
    inf_data_ouput_name:  output name of the inference data numpy array
    visualizeSampling:  visualize the sampling process
    device: device to run on
    """
    # check valid inf_data_ouput_name

    if is_valid_filename_practical(f"{inf_data_ouput_name}.npy") == False: print(f"Filename {inf_data_ouput_name} is not valid.")        
           
    # Check if dataset_path exists
    if os.path.exists(dataset_path):
        print(f"{dataset_path} exists.")
    else:
        print(f"{dataset_path} does not exist.")
        return

    # create dataset from file
    dataset = ImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        agent_pos_cutoff_position=agent_pos_cutoff_position,
        useSegmented=useSegmented
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # Load model nets

    nets = load_model_with_info(
        model_path, 
        device=device,
        lowdim_obs_dim=lowdim_obs_dim,
        vision_feature_dim=vision_feature_dim,
        obs_horizon=obs_horizon,
        action_dim=action_dim,
        useNoisePred=useNoisePred)

    if nets == 0:
        print("Model not loaded...quitting.")
        return
    
    
    
    # load robot sim
    sim = PushCubeEnv(width=img_width, height=img_height) # set width and height
    step = 1
    # Initialize video writer variables
    output_filename = 'output_video_seg.avi' if useSegmented else 'output_video.avi'
    

    # Initialize OpenCV VideoWriter
    if generateVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' or 'MJPG'
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (sim.width, sim.height))
    # Data storage
    images, states = [], []
    reset_counter = 0 # wie viele Resets?
    simIsDone = False
    infIsDone = False # is this inference loop done?

    done_counter = 0 # wie viele Erfolge?    
    inferenceTimes = []
    current_trajectory = [] # die aktuelle Lösungstrajektorie
    inferenceData = [] # die Daten der Lösungsschritte
    # Define the data type for the structured array
    dt = np.dtype([
        ('episode_num', np.int32),
        ('status', 'U10'),  # Unicode string up to 10 chars for "done"/"not done"
        ('success_rate', np.float32),
        ('inf_time_avg', np.float32),
        ('trajectory', object),  # For the trajectory list/array
        ('traj_length', np.int32),
        ('epoch_time', np.float32),
        ('cube_pos', np.float32, (3,))  # Assuming cube_pos is a 3D position
    ])    
    epochStartTime = 0
    epochTime = 0
    timeCount  = 0

    while infIsDone == False:
        # stack the last obs_horizon number of observations
        
        if epochStartTime == 0: epochStartTime = time.time() # start time of epoch
        
        if len(images) <2: # fill first 2 poses and images
            images.append(sim.get_camera_image(segmentation=useSegmented))
            states.append(sim.get_joint_angles())
            # record trajectory
            current_trajectory.append(sim.get_joint_angles())
        else:
            obs_images = images[-2:]
            #images = np.stack([x['image'] for x in obs_deque]) # get 2 images
            obs_poses = states[-2:]
            # agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(obs_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            
            nimages = np.array(obs_images).astype(np.float32)  # Convert to float
            nimages /= 255.0  # Normalize to [0, 1]
            nimages = torch.tensor(nimages).permute(0, 3, 1, 2) 
            #print(f"nimages.shape: {nimages.shape}")

            # device transfer
            nimages = nimages.to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            action_pred, inferenceTime = 0,0
            # Inference 
            action_pred, inferenceTime = infer_action(
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    stats=stats,
                    num_diffusion_iters=new_num_diffusion_iters,
                    ema_nets=nets, # give the nets!
                    useNoisePred=useNoisePred, # use noise prediction or sample?
                    DDPMScheduler_training_steps=DDPMScheduler_training_steps, # ddpm scheduler steps!                    
                    device=device,
                    pred_horizon=pred_horizon,
                    action_dim=action_dim,
                    sim=sim,
                    visualizeSampling=visualizeSampling
                )
            
            
            inferenceTimes.append(inferenceTime)
            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            
            # execute action_horizon number of steps without replanning           
            for i in range(len(action)):
                # stepping env
                # append states and images again
                images.append(sim.get_camera_image(segmentation=useSegmented))
                states.append(action[i])
                # print(f"action[{i}]: {action[i]}") 
                current_trajectory.append(action[i])
                
                #print(f"joint_angles: {joint_angles}")
                sim.set_joint_angles([action[i][0], action[i][1], action[i][2], action[i][3], action[i][4], action[i][5], action[i][6], 0 , 0, 0.02])
                if generateVideo:
                    video_writer.write(sim.get_camera_image(segmentation=useSegmented))  # Write frame to video
                # check if done in every step
                if sim.isDone(distanceThresholdCubetoGoal): 
                    simIsDone = True
                    epochTime = time.time() - epochStartTime # time of epoch
                time.sleep(1/fps)
                # Update timeCount
                timeCount += 1/fps
                # print(f"TimeCount: {timeCount:.2f}.")

        # Check for keyboard events
        keys = p.getKeyboardEvents()
        # Check if the 'r' key has been pressed to reset                    # geschafft       # gefailt
        if (ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED) or simIsDone or step > timeOutSetps or (timeCount >= timeCountLimit):
                      
            
            # handle success
            if simIsDone and timeCount < timeCountLimit:
                done_counter+=1
            else:
                epochTime = time.time() - epochStartTime # time of epoch
            reset_counter+=1 # before tracking
            # record inference data
            inferenceData.append((reset_counter, #number of inference episode
                                "done" if simIsDone else "not done", #number of successfully episodes
                                done_counter/reset_counter if reset_counter != 0 else -1,  #success rate
                                sum(inferenceTimes) / len(inferenceTimes), #inf time average
                                current_trajectory, #current_trajectory
                                len(current_trajectory), # length of trajectory
                                epochTime, #time of epoch
                                sim.get_cube_pos())) # red cube pos
            
                        
            print(f"Episode: {reset_counter}, done: {simIsDone}, Success Rate: {done_counter/reset_counter if reset_counter != 0 else -1}, Average Inference Time: {sum(inferenceTimes) / len(inferenceTimes)}, Epoch Time: {epochTime}, Time left: {((sum(record[6] for record in inferenceData)/reset_counter)* (inferenceEpochs - reset_counter))/60} min.")   
                  
            # reset
            simIsDone = False
            images, states, current_trajectory= [], [], []
            
            step = 0
            epochTime = 0
            epochStartTime = 0
            timeCount = 0
            sim.reset()
            
        
        step += 1
        # Check if the 'x' key has been pressed to exit
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED or reset_counter == inferenceEpochs:
            print("Exiting loop...") 
            # Convert to structured array
            inference_array = np.array(inferenceData, dtype=dt)  
            
            # save inference data                
            np.save(f'{inf_data_ouput_name}.npy', inference_array) 
            # Release the video writer
            if generateVideo:
                video_writer.release()
                cv2.destroyAllWindows()
            # break inference while loop
            infIsDone = True 
            sim.closeEnv()     
            break

if __name__ == "__main__":

    
    ''' 
    # run model with noise prediction
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "checkpoint_dir/42_seed_400ep_angles_ema_nets_model.pth", # distilled_model ema_student_2_steps.pth        
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 2, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "42_seed_400ep_angles_ema_nets_model.pth", 
        visualizeSampling=False,
        useNoisePred=True,
        fps=20
    )
    ''' 
    # test progressively distilled model with 1 inference step and weight balance = 0.2 against normal noise prediction model
    # run model with sample prediction
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "checkpoint_dir/ema_student_3_steps400eps.pth", # distilled_model ema_student_2_steps.pth        
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 4, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "ema_student_3_steps400eps_inf4.pth", 
        visualizeSampling=False,
        useNoisePred=False,
        fps=20 # maximal ausgeführte robo actions per second
    )
    '''  
    # normal noise prediction model
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "checkpoint_dir/42_seed_400ep_angles_ema_nets_model.pth", # distilled_model ema_student_2_steps.pth        
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 1, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "42_seed_400ep_angles_ema_nets_model.pth", 
        visualizeSampling=False,
        useNoisePred=True,
        fps=20
    )
     
    
    
    run_simulation(        
        lowdim_obs_dim = 3,
        action_dim = 7,# dim of actions 
        model_path = "checkpoint_dir/ema_student_3_steps.pth", 
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 3, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "seed_42_ema_student_3_steps_inf_2.pth", 
        visualizeSampling=False,
        useNoisePred=False
    )
    '''
    
    
    
   
    
    




