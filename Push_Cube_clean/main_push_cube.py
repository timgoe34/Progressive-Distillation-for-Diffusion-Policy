
# import dataset functions
from dataset import *
# import u net functions
from network import *
# import vision encoder functions
from vision_encoder import *
# import standard training
from train_baseline import *
# import distiller functions
from distiller_functions import *
# import load and run functions
from load_and_run import *
# import utils
from utils import *
import gdown                    # for downloading from google drive


# Tabelle 5.5

def push_fast_angles_replicate_original_model():
    # this function first trains a normal model on the 400 angles dataset and then runs it for 1000 times. It will repllicate the data for the noise residual model evaluated on one step

    baseline_train_and_save_model(
        dataset_path="angles_data_storage_400ep.zarr.zip",        
        lowdim_obs_dim = 7,  # robo angles is 7-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="models",
        name_of_model="400ep_angles_ema_nets_model",
        seed=1000,   
        agent_pos_cutoff_position = 7 
    )

    # normal noise prediction model
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "models/1000_seed_400ep_angles_ema_nets_model.pth", # distilled_model ema_student_2_steps.pth        
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr.zip"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 1, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "infData/42_seed_400ep_angles_ema_nets_model.pth", 
        visualizeSampling=False,
        useNoisePred=True,
        fps=20
    )

def push_fast_angles_replicate_dest3_step():
    # this function first trains a normal model on the 400 angles dataset and then runs it for 1000 times. It will repllicate the data for the noise residual model evaluated on one step

    
    baseline_train_and_save_model_sample(
        dataset_path="angles_data_storage_400ep.zarr.zip",        
        lowdim_obs_dim = 7,  # angles is 7-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="models",
        name_of_model="400ep_angles_ema_nets_model_sample",
        seed=42,
        agent_pos_cutoff_position = 7 
    )
    

    progressive_distillation_training(
        trained_nets=load_model_with_info(
            model_path="models/42_seed_400ep_angles_ema_nets_model_sample.pth", 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lowdim_obs_dim=7,
            vision_feature_dim=512,
            obs_horizon=2,
            action_dim=7,
            useNoisePred=False), 
        dataloader=create_dataloader(
            dataset_path="angles_data_storage_400ep.zarr.zip",
            agent_pos_cutoff_position=7),
        checkpoint_saving_dir=os.path.join(os.getcwd(), 'models'),
        num_epochs=15,
        initial_steps=100,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epoch_loss_threshold=0, # 0.005
        useEma = True,
        optimizer_weight_decay = 1e-6, # 0.1 oder 1e-6, geringer für mehr Generalisierbarkeit
        loss_weight_balance = 0.2,
        model_name="400eps"
    ) 

    # run inference for model for Dest3-3 with 1 step
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "models/ema_student_1_steps400eps.pth",    
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr.zip"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 1, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "ema_student_1_steps400eps_inf1.pth", 
        visualizeSampling=False,
        useNoisePred=False,
        fps=20 # maximal ausgeführte robo actions per second
    )

    # run inference for model for Dest3-3 with 2 steps
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "models/ema_student_3_steps400eps.pth",    
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr.zip"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 2, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "ema_student_3_steps400eps_inf2.pth", 
        visualizeSampling=False,
        useNoisePred=False,
        fps=20 # maximal ausgeführte robo actions per second
    )

    # run inference for model for Dest3-3 with 4 steps
    run_simulation_with_angles(            
        lowdim_obs_dim = 7,        
        agent_pos_cutoff_position= 7,
        action_dim = 7,# dim of actions 
        model_path = "models/ema_student_3_steps400eps.pth",    
        dataset_path=os.path.join(os.getcwd(), "angles_data_storage_400ep.zarr.zip"),
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 4, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "ema_student_3_steps400eps_inf4.pth", 
        visualizeSampling=False,
        useNoisePred=False,
        fps=20 # maximal ausgeführte robo actions per second
    )


# Tabelle 5.4 - replicate original and dest3 as dest3 is most promosing

def push_cube_replicate_original_model():
    # this function first trains a normal model on the data_storage.zarr.zip dataset and then runs it for 1000 times. It will repllicate the data for the noise residual model evaluated on two steps

    
    baseline_train_and_save_model(
        dataset_path="data_storage.zarr.zip",        
        lowdim_obs_dim = 3,  # agent_pos is 3-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="models",
        name_of_model="ee_ema_nets_model",
        seed=42
    )
    
    # normal noise prediction model
    run_simulation(    
        dataset_path = "data_storage.zarr.zip",
        lowdim_obs_dim = 3,
        action_dim = 7,# dim of actions 
        model_path = "models/42_seed_ee_ema_nets_model.pth", 
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 2, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "42_seed_ee_ema_nets_model.pth", 
        visualizeSampling=False,
        useNoisePred=True
    )

def push_cube_replicate_dest3():
    # this function first trains a normal model on the data_storage.zarr.zip dataset and then runs it for 1000 times. It will repllicate the data for the noise residual model evaluated on two steps

    
    baseline_train_and_save_model_sample(
        dataset_path="data_storage.zarr.zip",        
        lowdim_obs_dim = 3,  # agent_pos is 3-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="models",
        name_of_model="ee_ema_nets_model_sample",
        seed=42
    )

    progressive_distillation_training(
        trained_nets=load_model_with_info(
            model_path="models/42_seed_ee_ema_nets_model_sample.pth", 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lowdim_obs_dim=3,
            vision_feature_dim=512,
            obs_horizon=2,
            action_dim=7,
            useNoisePred=False), 
        dataloader=create_dataloader(
            dataset_path="data_storage.zarr.zip",
            agent_pos_cutoff_position=3),
        checkpoint_saving_dir=os.path.join(os.getcwd(), 'models'),
        num_epochs=15,
        initial_steps=100,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epoch_loss_threshold=0, # 0.005
        useEma = True,
        optimizer_weight_decay = 1e-6, # 0.1 oder 1e-6, geringer für mehr Generalisierbarkeit
        loss_weight_balance = 0.5,
        model_name="ee"
    )
   
    
    # sample prediction model
    run_simulation(  
        dataset_path = "data_storage.zarr.zip",      
        lowdim_obs_dim = 3,
        action_dim = 7,# dim of actions 
        model_path = "models/ema_student_3_stepsee.pth", 
        useSegmented = False, 
        inferenceEpochs = 1000, 
        new_num_diffusion_iters = 2, 
        DDPMScheduler_training_steps = 100, 
        inf_data_ouput_name = "seed_42_ee_ema_student_3_stepsee_inf_2.pth", 
        visualizeSampling=False,
        useNoisePred=False
    )

if __name__ == "__main__":

    ## Download datasets 
    ee_data_path = "data_storage.zarr.zip"
    if not os.path.isfile(ee_data_path):
        print("Downloading dataset for EE-Positions") 
        id = "1-WmmekuK7Rd7J_IGEMyH_rwE517DbTuX&confirm=t" 
        gdown.download(id=id, output=ee_data_path, quiet=False) 

    angles_data_path = "angles_data_storage_400ep.zarr.zip"
    if not os.path.isfile(angles_data_path):
        print("Downloading dataset for angles") 
        id = "1HcecddDJYHoUVNitxkr7qDbguLehCXPK&confirm=t" 
        gdown.download(id=id, output=angles_data_path, quiet=False)
    
    ## Replicate Push Env Results

    # push_fast_angles_replicate_original_model() # works

    # push_fast_angles_replicate_dest3_step() # works

    # push_cube_replicate_original_model() # works
    
    push_cube_replicate_dest3() # works