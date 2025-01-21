"""
This python file does:

- defines a function "baseline_train_and_save_model" to train a diffusion model to predict noise residuals
- defines a function "baseline_train_and_save_model_sample" to train a diffusion model to predict denoised samples direktly

"""

# IMPORTS

import numpy as np
import torch
import os
from tqdm import tqdm
import torch
import time
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# import dataset functions
from dataset import *
# import u net functions
from network import *
# import vision encoder functions
from vision_encoder import *




def baseline_train_and_save_model(
    dataset_path: str,
    lowdim_obs_dim,  # agent_pos is 3-dimensional
    action_dim,  # we have 7 inputs
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,   
    agent_pos_cutoff_position: int = 3, 
    vision_feature_dim: int = 512, # ResNet18 output dim is 512    
    useSegmented: bool = False,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    num_diffusion_iters: int = 100,
    checkpoint_dir: str = "checkpoint_dir",
    name_of_model: str = "ema_nets_model",
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42,  

):
    """
    Function to train a model using the given parameters and save the trained model.

    Args:
    - dataset_path (str): Path to the dataset.
    - pred_horizon (int): Prediction horizon.
    - obs_horizon (int): Observation horizon.
    - action_horizon (int): Action horizon.
    - useSegmented (bool): Whether to use segmented data.
    - batch_size (int): Batch size for training.
    - num_epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for optimizer.
    - weight_decay (float): Weight decay for optimizer.
    - num_diffusion_iters (int): Number of diffusion iterations.
    - checkpoint_dir (str): Directory where the trained model will be saved.
    - name_of_model (str): the name of the model saved in checkpoint_dir.
    - seed (int): number to make the training process more reproducable --> model_checkpoint_path = os.path.join(checkpoint_dir, f"{seed}_seed_{name_of_model}.pth")
    """

    '''
    # Example usage:
    baseline_train_and_save_model(
        dataset_path="data_storage.zarr",
        lowdim_obs_dim = 3,  # agent_pos is 3-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="checkpoint_dir",
        name_of_model="ema_nets_model_1",
        seed=42
    )
    '''
    start_time = time.time()
    # Check if dataset exists
    if not os.path.exists(os.path.join(os.getcwd(), dataset_path)):
        raise FileNotFoundError(f"{dataset_path} not available!")

    # Setzen des globalen Seeds für Reproduzierbarkeit
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # Optional: für vollständige Reproduzierbarkeit
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Seed zu Beginn der Funktion setzen
    set_seed(seed)

    # Create dataset from file
    dataset = ImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        useSegmented=useSegmented,
        agent_pos_cutoff_position=agent_pos_cutoff_position
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate
    )

    # Create ResNet18 encoder
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    
    obs_dim = vision_feature_dim + lowdim_obs_dim
    

    # Create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    # Networks dictionary
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # Diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Set device    
    _ = nets.to(device)

    # Exponential Moving Average (EMA) model
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Cosine LR scheduler with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_loss = 0
    # Training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Device transfer
                    nimage = nbatch['image'][:, :obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # Normalize images
                    nimage = nimage.float() / 255.0
                    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)

                    # Concatenate vision feature and low-dim observations
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    # Sample noise and timesteps
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # Add noise to actions (forward diffusion)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # Predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # Compute L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # Backpropagate and optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Step lr scheduler
                    lr_scheduler.step()

                    # Update EMA model
                    ema.step(nets.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            final_loss = np.mean(epoch_loss)

    # Copy EMA weights to the model
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    # Save the trained model
    #model_checkpoint_path = os.path.join(checkpoint_dir, f"{seed}_seed_{name_of_model}.pth")
    print(f"Saving model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {final_loss}\n')
            
    #torch.save(ema_nets.state_dict(), )
    save_path = os.path.join(checkpoint_dir, f'{seed}_seed_{name_of_model}.pth')
    torch.save({
        'model_state_dict': ema_nets.state_dict(),
        'training_info': {
            'training_time': time.time() - start_time,
            'current_mean_loss': final_loss,
            'prediction_type': "epsilon"
        }
    }, save_path)
    print(f"Model saved to {save_path}.")

def baseline_train_and_save_model_sample(
    dataset_path: str,
    lowdim_obs_dim,  # agent_pos is 3-dimensional
    action_dim,  # we have 7 inputs
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8, 
    agent_pos_cutoff_position: int = 3, # if the agent pos looks like this [1,2,3,4,5,6,7] , if this is "3" then it will only take [1,2,3] from the agent pos vector to learn  
    vision_feature_dim: int = 512, # ResNet18 output dim is 512    
    useSegmented: bool = False,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    num_diffusion_iters: int = 100,
    checkpoint_dir: str = "checkpoint_dir",
    name_of_model: str = "ema_nets_model",
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 42,  

):
    print("Training model that predicts sample instead of noise residuals.")
    """
    Function to train a model using the given parameters and save the trained model.

    Args:
    - dataset_path (str): Path to the dataset.
    - pred_horizon (int): Prediction horizon.
    - obs_horizon (int): Observation horizon.
    - action_horizon (int): Action horizon.
    - useSegmented (bool): Whether to use segmented data.
    - batch_size (int): Batch size for training.
    - num_epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for optimizer.
    - weight_decay (float): Weight decay for optimizer.
    - num_diffusion_iters (int): Number of diffusion iterations.
    - checkpoint_dir (str): Directory where the trained model will be saved.
    - name_of_model (str): the name of the model saved in checkpoint_dir.
    - seed (int): number to make the training process more reproducable --> model_checkpoint_path = os.path.join(checkpoint_dir, f"{seed}_seed_{name_of_model}.pth")
    """
    start_time = time.time()
    # Check if dataset exists
    if not os.path.exists(os.path.join(os.getcwd(), dataset_path)):
        raise FileNotFoundError(f"{dataset_path} not available!")

    # Setzen des globalen Seeds für Reproduzierbarkeit
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # Optional: für vollständige Reproduzierbarkeit
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Seed zu Beginn der Funktion setzen
    set_seed(seed)

    # Create dataset from file
    dataset = ImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        agent_pos_cutoff_position = agent_pos_cutoff_position,
        useSegmented=useSegmented
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=custom_collate
    )

    # Create ResNet18 encoder
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    
    obs_dim = vision_feature_dim + lowdim_obs_dim
    

    # Create network object
    sample_pred_net = ConditionalUnet1D_sample(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    # Networks dictionary
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'sample_pred_net': sample_pred_net
    })

    # Diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='sample'
    )

    # Set device    
    _ = nets.to(device)

    # Exponential Moving Average (EMA) model
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Cosine LR scheduler with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_loss = 0
    # Training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Device transfer
                    nimage = nbatch['image'][:, :obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # Normalize images
                    nimage = nimage.float() / 255.0
                    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)

                    # Concatenate vision feature and low-dim observations
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    # Sample noise and timesteps
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # Add noise to actions (forward diffusion)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # Predict the noise sample
                    noisy_sample = sample_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # Compute L2 loss
                    loss = nn.functional.mse_loss(noisy_sample, naction)

                    # Backpropagate and optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Step lr scheduler
                    lr_scheduler.step()

                    # Update EMA model
                    ema.step(nets.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            final_loss = np.mean(epoch_loss)

    # Copy EMA weights to the model
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    # Save the trained model
    print(f"Saving model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {final_loss}\n')
            
    #torch.save(ema_nets.state_dict(), )
    save_path = os.path.join(checkpoint_dir, f'{seed}_seed_{name_of_model}.pth')
    torch.save({
        'model_state_dict': ema_nets.state_dict(),
        'training_info': {
            'training_time': time.time() - start_time,
            'current_mean_loss': final_loss,
            'prediction_type': "sample"
        }
    }, save_path)
    print(f"Model saved to {save_path}.")


if __name__ == "__main__":

    '''
    baseline_train_and_save_model(
        dataset_path="angles_data_storage_400ep.zarr",        
        lowdim_obs_dim = 7,  # agent_pos is 7-dimensional
        agent_pos_cutoff_position = 7,
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="checkpoint_dir",
        name_of_model="400ep_angles_ema_nets_model",
        seed=42
    )

    baseline_train_and_save_model_sample(
        dataset_path="angles_data_storage_400ep.zarr",        
        lowdim_obs_dim = 7,  # agent_pos is 7-dimensional
        agent_pos_cutoff_position = 7,
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="checkpoint_dir",
        name_of_model="400ep_angles_ema_nets_model_sample",
        seed=42
    )
    '''
    
    # train with sample
    ''' 
    baseline_train_and_save_model_sample(
        dataset_path="data_storage.zarr",        
        lowdim_obs_dim = 3,  # agent_pos is 3-dimensional
        action_dim = 7,  # we have 7 inputs
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_diffusion_iters=100,
        checkpoint_dir="checkpoint_dir",
        name_of_model="ema_nets_model_sample",
        seed=42
    )
    '''
   
