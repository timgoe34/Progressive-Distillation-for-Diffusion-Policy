"""
This python file does:
- defines a couple of progresive distillation algorithms and also configs for them
- defines some helper functions like 
    prepare_data to form latent observation data from multiple inputs and also
    destillation_loss to calcualte the loss

"""

# IMPORTS

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from tqdm.auto import tqdm
from copy import deepcopy
from datetime import datetime
import time
# import dataset functions
from dataset import *
# import u net functions
from network import *
# import vision encoder functions
from vision_encoder import *
# import utils for loading models, checking file paths, et.c
from utils import *



def prepare_data(nbatch, obs_horizon, nets, device):
    """
    build latent observations tensor from batch with given nets, obs_horizon and on device
    Args:
        nbatch: training data batch.
        obs_horizon: length of observed horizon steps (usually 16).
        nets: nets with vison encoder to use.
        device: device to use.
    """
    # Prepare data
    nimage = nbatch['image'][:,:obs_horizon].to(device)
    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
    # Normalize and process vision features
    nimage = nimage.float() / 255.0
    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1)) # maybe change with student nets!
    image_features = image_features.reshape(*nimage.shape[:2], -1)
    # Combine vision features with agent positions
    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
    return( obs_features.flatten(start_dim=1) )

def progressive_distillation_training(
    dataloader, 
    trained_nets, 
    checkpoint_saving_dir, 
    num_epochs, 
    initial_steps, 
    device, 
    obs_horizon = 2, 
    epoch_loss_threshold = 0, 
    useEma = False, 
    optimizer_weight_decay = 1e-6,
    loss_weight_balance = 0.5, # how are the loss terms weighted? default to 50_50
    model_name = ""
):
    """
    Progressive Distillation Algorithm for training a student model with fewer diffusion steps.

    Args:
        dataloader: PyTorch DataLoader for training data.
        nets: Dictionary of model components (e.g., vision_encoder, noise_pred_net).
        noise_scheduler: Noise scheduler (e.g., DDPMScheduler).
        useEma: save ema model or normal model?
        checkpoint_saving_dir: Directory to save checkpoints.
        num_epochs: Number of epochs per stage.
        initial_steps: Initial number of diffusion steps for the teacher model.
        device: Torch device for training.
        epoch_loss_threshold: if this loss is reached during one distillation epoch, then the next epoch starts! if left at 0, it wont stop early
        optimizer_weight_decay: weight_decay of optimizer # 1e-6 or maybe 0.1?
    """
    # Note that EMA parameter are not optimized
    

    os.makedirs(checkpoint_saving_dir, exist_ok=True)
    teacher_nets = trained_nets # assign pretrained nets
    teacher_steps = initial_steps
    student_steps = teacher_steps // 2 #  1
    # ema model
    ema_model = EMAModel(
        parameters=trained_nets.parameters(),
        power=0.75)
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type= 'sample' # sample for denoised sample
    )
    start_time = time.time()
    while student_steps >= 1:  # Progressive halving
        print(f"Distillation Stage: Teacher {teacher_steps} steps -> Student {student_steps} steps")
        # init student from teacher
        student_nets = deepcopy(teacher_nets) # added deepcopy
        opt = torch.optim.AdamW(
            params=student_nets.parameters(),
            lr=1e-4, 
            weight_decay = optimizer_weight_decay
        ) 
        
        distill_mean_loss = 0
        with tqdm(range(num_epochs), desc=f'Epoch (Steps {student_steps})') as tglobal:
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
                        image_features = teacher_nets['vision_encoder'](nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(*nimage.shape[:2], -1)

                        # Concatenate vision feature and low-dim observations
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        
                        # Sample noise and timesteps
                        noise = torch.randn(naction.shape, device=device)
                        timesteps = torch.randint(
                            0, teacher_steps,
                            (B,), device=device
                        ).long()
                        # Add noise to actions (forward diffusion)
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # Predict the noise sample
                        student_noisy_sample = student_nets['sample_pred_net']( noisy_actions, timesteps//2, global_cond=obs_cond)
                        teacher_noisy_sample = teacher_nets['sample_pred_net']( noisy_actions, timesteps, global_cond=obs_cond)

                        # Compute L2 loss
                        loss = loss_weight_balance * F.mse_loss(student_noisy_sample, naction) + (1-loss_weight_balance) * F.mse_loss(student_noisy_sample, teacher_noisy_sample)

                        # Backpropagate and optimize
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                        # lr_scheduler.step() maybe
                        # try with kl divergence loss                       
                        #kl_loss = nn.KLDivLoss(reduction='batchmean')   
                        ema_model.step(student_nets.parameters())

                        # Log and track progress
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                    # early stop if loss is below threshold
                    distill_mean_loss = np.mean(epoch_loss)
                    if np.mean(epoch_loss) <= epoch_loss_threshold and np.mean(epoch_loss) != 0 and epoch_loss_threshold != 0:
                        print(f"\nStopping early at epoch {epoch_idx} with loss: {np.mean(epoch_loss):.5f} <= {epoch_loss_threshold}")
                        break  

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                

        # Save checkpoints
        if useEma:            
            # move to EMA weights
            ema_nets = student_nets
            ema_model.copy_to(ema_nets.parameters())
            print(f"Saving ema model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {distill_mean_loss}\n',
                f'  current_inf_steps: {student_steps}\n')
            
            
            save_path = os.path.join(checkpoint_saving_dir, f'ema_student_{student_steps}_steps{model_name}.pth')
            torch.save({
                'model_state_dict': ema_nets.state_dict(),
                'training_info': {
                    'training_time': time.time() - start_time,
                    'current_mean_loss': distill_mean_loss,
                    'current_inf_steps': student_steps,
                }
            }, save_path)
        else:
            
            print(f"Saving normal model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {distill_mean_loss}\n',
                f'  current_inf_steps: {student_steps}\n')
            
            
            save_path = os.path.join(checkpoint_saving_dir, f'student_{student_steps}_steps{model_name}.pth')
            torch.save({
                'model_state_dict': student_nets.state_dict(),
                'training_info': {
                    'training_time': time.time() - start_time,
                    'current_mean_loss': distill_mean_loss,
                    'current_inf_steps': student_steps,
                }
            }, save_path)
        # assign teacher from student
        teacher_nets = deepcopy(student_nets) 
        # Halve steps for the next stage
        teacher_steps = student_steps
        student_steps //= 2


'''
Weitere Versuche:
- kopiere den vision encoder vom Teacher zum Student nach dem Training
- generiere obs_cond mit den zwei unterschiedlichen Vision Encoders der Netze Student and Teacher --> loss bleibt bei ca.  0.28 für decay weights 0.1 - warum?


progressive_distillation_training(
    trained_nets=load_model_with_info(
        model_path="checkpoint_dir/42_seed_400ep_angles_ema_nets_model_sample.pth", 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lowdim_obs_dim=7,
        vision_feature_dim=512,
        obs_horizon=2,
        action_dim=7,
        useNoisePred=False), 
    dataloader=create_dataloader(
        dataset_path="angles_data_storage_400ep.zarr",
        agent_pos_cutoff_position=7),
    checkpoint_saving_dir=os.path.join(os.getcwd(), 'checkpoint_dir'),
    num_epochs=15,
    initial_steps=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epoch_loss_threshold=0, # 0.005
    useEma = True,
    optimizer_weight_decay = 1e-6, # 0.1 oder 1e-6, geringer für mehr Generalisierbarkeit
    loss_weight_balance = 0.2,
    model_name="400eps"
) # Training time: 17.68 min
'''
"""

progressive_distillation_training(
    trained_nets=load_model_with_info(
        model_path="checkpoint_dir/42_seed_ema_nets_model_sample.pth", 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lowdim_obs_dim=3,
        vision_feature_dim=512,
        obs_horizon=2,
        action_dim=7,
        useNoisePred=False), 
    dataloader=create_dataloader(),
    checkpoint_saving_dir=os.path.join(os.getcwd(), 'checkpoint_dir'),
    num_epochs=15,
    initial_steps=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epoch_loss_threshold=0, # 0.005
    useEma = True,
    optimizer_weight_decay = 1e-6, # 0.1 oder 1e-6, geringer für mehr Generalisierbarkeit
    loss_weight_balance = 0.5
)
Diffusion Model Distillation Progress:
+------------+---------------+---------------+--------+---------------+-----------+-------------------------+
| Stage     | Teacher Steps | Student Steps | Epochs | Avg Time/Epoch| Final Loss| Model Saved             |
+------------+---------------+---------------+--------+---------------+-----------+-------------------------+
| 1         | 100           | 50            | 15     | 12.80s        | 0.000214  | ema_student_50_steps.pth|
| 2         | 50            | 25            | 15     | 11.74s        | 0.00021   | -                      |
| 3         | 25            | 12            | 15     | 11.04s        | 0.000157  | ema_student_12_steps.pth|
| 4         | 12            | 6             | 15     | 11.02s        | 0.000121  | ema_student_6_steps.pth |
| 5         | 6             | 3             | 15     | 11.04s        | 5.84e-5   | ema_student_3_steps.pth | 11.65 miin
| 6         | 3             | 1             | 15     | 13.73s        | 7.24e-5   | ema_student_1_steps.pth |
+------------+---------------+---------------+--------+---------------+-----------+-------------------------+

Distillation Process Notes:
- Consistent 15 epochs per stage
- Gradual reduction of diffusion steps
- Maintained low loss throughout compression
"""


def distillation_loss(student_pred, teacher_pred, true_noise, 
                      alpha=0.5, temperature=2.0, useTemp=True):
    """
    Compute combined hard and soft distillation loss
    
    Args:
        student_pred: Student's noise prediction (tensor)
        teacher_pred: Teacher's noise prediction (tensor or tuple)
        true_noise: Ground truth noise
        alpha: Balance between hard and soft losses
        temperature: Soft target temperature
    """
    # Handle case where teacher_pred might be a tuple
    if isinstance(teacher_pred, tuple):
        teacher_pred = teacher_pred[0]  # Take first prediction if tuple
    
    # Hard loss (original task loss)
    hard_loss = F.mse_loss(student_pred, true_noise)
    
    # Soft loss (knowledge distillation loss)
    soft_loss = F.mse_loss(
        student_pred / temperature, 
        teacher_pred / temperature
    ) if useTemp else F.mse_loss(
        student_pred, 
        teacher_pred
    )
    
    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss 
    return total_loss

def progressive_distillation_temperature( 
    teacher_nets, 
    dataloader, 
    checkpoint_saving_dir,
    num_distillation_steps=2, 
    initial_temperature=2.0,
    epochs = 50,
    device=torch.device('cuda'),
    useTemp=True,
    useEma=True,
    obs_horizon=2,
    action_dim=7,
    obs_dim=515
):
    """
    Progressively distill the diffusion model with explicit device management
    
    Args:
        teacher_nets: Original teacher networks
        dataloader: Training data
        num_distillation_steps: Number of progressive distillation iterations
        initial_temperature: Starting temperature for knowledge distillation
        device: Target device for computations
        useTemp: use dynamic temperature scaling
        useEma: save ema model or normal model?
        checkpoint_saving_dir: directory where checkpoints should be saved
    """
    start_time = time.time()
    # Ensure teacher is on the specified device
    current_teacher = teacher_nets.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type= 'epsilon' # sample for denoised sample, epsilon for predicted noise
    )
    
    for step in range(num_distillation_steps):
        
        # Create student network on the same device
        print(f"Distillation step: {step}")
        student_nets = nn.ModuleDict({
            'vision_encoder': create_smaller_encoder(current_teacher['vision_encoder']).to(device),
            'noise_pred_net': StudentNetwork(
                current_teacher['noise_pred_net'], action_dim, obs_dim, obs_horizon #,distillation_temperature=initial_temperature / (step + 1)                
            ).to(device)
        })
        # ema model
        ema_model = EMAModel( parameters=student_nets.parameters(), power=0.75)
        # Student training loop (similar to original training)
        optimizer = torch.optim.AdamW(
            params=student_nets.parameters(),
            lr=1e-4, weight_decay=1e-6
        )
        
        for epoch in range(epochs):  # Reduced epoch count for distillation
            epoch_losses = []
            for batch in dataloader:
                # Explicit device transfer
                nimage = batch['image'][:,:obs_horizon].to(device)
                nagent_pos = batch['agent_pos'][:,:obs_horizon].to(device)
                naction = batch['action'].to(device)
                
                # Encode images
                nimage = nimage.float() / 255.0
                image_features = student_nets['vision_encoder'](nimage.flatten(end_dim=1))
                image_features = image_features.reshape(*nimage.shape[:2],-1)
                
                # Prepare conditions
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                
                # Sample noise
                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (naction.shape[0],), device=device
                ).long()
                
                # Add noise
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                
                # Ensure everything is on the correct device
                noisy_actions = noisy_actions.to(device)
                timesteps = timesteps.to(device)
                obs_cond = obs_cond.to(device)
                
                # Get predictions
                student_pred, teacher_pred = student_nets['noise_pred_net'](
                    noisy_actions, timesteps, global_cond=obs_cond
                )
                
                # Compute loss
                loss = distillation_loss(
                    student_pred, 
                    teacher_pred, 
                    noise, 
                    alpha=0.5, 
                    temperature=initial_temperature / (step + 1),
                    useTemp = useTemp
                )
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                #print(f"Loss: {loss.item()}")
            print(f"Epoch: {epoch}. mean loss: {np.mean(epoch_losses):.5f}")
        
        # Update teacher for next iteration
        current_teacher = student_nets

    end_time = (time.time() - start_time)
    print(f"Distillation time: {end_time:.3f} s")
    # Determine model type based on EMA and temperature usage
    if useEma:
        model_type = "distilled_model_ema_with_temp" if useTemp else "distilled_model_ema_no_temp"
        ema_nets = current_teacher
        # Move to EMA weights
        ema_model.copy_to(ema_nets.parameters())
        # Save the distilled model state
        model_checkpoint_path = os.path.join(checkpoint_saving_dir, f"{model_type}.pth")
        torch.save(ema_nets.state_dict(), model_checkpoint_path)
        print(f"Distilled ema model saved to {model_checkpoint_path} \n")
    else:
        model_type = "distilled_model_no_ema_with_temp" if useTemp else "distilled_model_no_ema_no_temp"        
        # Save the distilled model state
        model_checkpoint_path = os.path.join(checkpoint_saving_dir, f"{model_type}.pth")
        torch.save(current_teacher.state_dict(), model_checkpoint_path)
        print(f"Distilled normal model saved to {model_checkpoint_path} \n")
    


# Modify StudentNetwork to handle device consistency
class StudentNetwork(ConditionalUnet1D):
    def __init__(self, teacher_network, action_dim, obs_dim, obs_horizon): 
        super().__init__(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )
        # Ensure teacher is on the same device
        self.teacher = teacher_network.to(next(self.parameters()).device)
        self.teacher.eval()  # Freeze teacher
        #self.temperature = distillation_temperature

    def forward(self, noisy_actions, timesteps, global_cond):
        # Ensure all inputs are on the same device
        device = next(self.parameters()).device
        noisy_actions = noisy_actions.to(device)
        timesteps = timesteps.to(device)
        global_cond = global_cond.to(device)
        
        # Student's primary prediction
        student_pred = super().forward(noisy_actions, timesteps, global_cond)
        
        # Soft distillation during training
        with torch.no_grad():
            teacher_pred = self.teacher(noisy_actions, timesteps, global_cond)
        
        return student_pred, teacher_pred

def create_smaller_encoder(original_encoder):
    """
    Create a smaller vision encoder through layer pruning or reduction
    Implement specific reduction strategy based on your architecture
    """
    # Example: Reduce channels, remove layers, etc.
    reduced_encoder = original_encoder
    return reduced_encoder


'''
progressive_distillation_temperature(
    teacher_nets=load_model_nets(model_path="checkpoint_dir/ema_nets_model.pth"), 
    dataloader=create_dataloader(), 
    checkpoint_saving_dir=os.path.join(os.getcwd(), 'checkpoint_dir'),
    num_distillation_steps=2, 
    initial_temperature=2.0,
    useTemp=False,
    epochs = 50,
    useEma=False
)
'''



class ProgressiveDistillerConfig:
    def __init__(
        self, 
        original_nets,  # Original networks dictionary
        dataloader,     # Training dataloader
        checkpoint_saving_dir,
        dest_config, # destillation config
        model_name_postfix, # will be used to name intermediate model net saves f"distilled_model_comb_temp_infsteps_{config['num_diffusion_iters']}_steps_{self.model_name_postfix}.pth"
        obs_horizon=2, # obs horizon
        action_dim=7,
        obs_dim = 515,
        device=torch.device('cuda')  
    ):
        self.original_nets = original_nets.to(device)
        self.dataloader = dataloader
        self.device = device
        self.checkpoint_saving_dir = checkpoint_saving_dir
        self.action_dim =action_dim
        self.obs_dim =obs_dim
        self.obs_horizon = obs_horizon
        self.model_name_postfix = model_name_postfix

        # create normal noise scheduler
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # Distillation configurations
        self.distillation_configs = dest_config
    
    def create_smaller_encoder(self, original_encoder):
        """
        Dynamisch Encoder verkleinern
        
        smaller_encoder = copy.deepcopy(original_encoder)
        
        # Reduziere Kanalkapazität
        for name, module in smaller_encoder.named_children():
            if hasattr(module, 'out_channels'):
                module.out_channels = max(module.out_channels // 2, 16)
        """
        return original_encoder
    
    def distillation_loss(
        self, 
        student_pred, 
        teacher_pred, 
        true_noise, 
        temperature=2.0, 
        alpha=0.5 # nur softloss
    ):
        """
        Kombinierter Verlust für Wissenstransfer
        """
        # Sicherstellen, dass teacher_pred ein Tensor ist
        if isinstance(teacher_pred, tuple):
            teacher_pred = teacher_pred[0]
        
        # Hard Loss (originale Aufgabe)
        hard_loss = F.mse_loss(student_pred, true_noise)
        
        # Soft Loss mit Temperatur-Skalierung
        soft_loss = F.mse_loss(
            student_pred / temperature, 
            teacher_pred / temperature
        )
        
        # Kombinierter Verlust
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def distill(self):
        """
        Hauptmethode für progressive Distillation
        """
        start_time = time.time()
        current_nets = self.original_nets # deepcopy(self.original_nets)#         
        
        for config in self.distillation_configs:
            # Kleinere Encoder erstellen
            smaller_vision_encoder = self.create_smaller_encoder(
                current_nets['vision_encoder']
            )
            
            # Student-Netzwerk initialisieren
            student_nets = nn.ModuleDict({
                'vision_encoder': smaller_vision_encoder.to(self.device),
                'noise_pred_net': StudentNetwork_half_steps(
                    current_nets['noise_pred_net'], self.action_dim, self.obs_dim, self.obs_horizon
                ).to(self.device)
            })
            
            # Optimierer und Scheduler
            optimizer = torch.optim.AdamW(
                params=student_nets.parameters(),
                lr=config['lr'], 
                weight_decay=1e-6
            )
            currentDistillStep_mean_loss = 0
            # Training
            for epoch in range(config['epochs']):
                epoch_losses = []
                
                for batch in self.dataloader:
                    # Daten vorbereiten
                    nimage = batch['image'][:,:self.obs_horizon].to(self.device)
                    nagent_pos = batch['agent_pos'][:,:self.obs_horizon].to(self.device)
                    naction = batch['action'].to(self.device)
                    
                    # Bildmerkmale extrahieren
                    nimage = nimage.float() / 255.0
                    image_features = student_nets['vision_encoder'](
                        nimage.flatten(end_dim=1)
                    )
                    image_features = image_features.reshape(*nimage.shape[:2],-1)
                    
                    # Bedingungen vorbereiten
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    
                    # Rauschen samplen
                    noise = torch.randn(naction.shape, device=self.device)
                    timesteps = torch.randint(
                        0, config['num_diffusion_iters'],
                        (naction.shape[0],), device=self.device
                    ).long()
                    
                    # Rauschen hinzufügen
                    noisy_actions = self.noise_scheduler.add_noise(
                        naction, noise, timesteps
                    )                    
                    # Vorhersagen
                    student_pred, teacher_pred = student_nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond
                    )
                    # Verlust berechnen                    
                    loss = self.distillation_loss(
                        student_pred, 
                        teacher_pred, 
                        noise, 
                        temperature=config['temperature']
                    )                   
                    # Optimieren
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                # Epochen-Logging
                print(f"Config: {config}, " 
                      f"Epoch: {epoch}, "
                      f"Avg Loss: {np.mean(epoch_losses):.6f}")
                currentDistillStep_mean_loss = np.mean(epoch_losses)
            # Aktualisiere Teacher für nächste Iteration
            current_nets = student_nets # deepcopy(student_nets)#
            #del student_nets # delete this student net its not needed
            #torch.cuda.empty_cache()

            # Save the distilled temperature model state
            end_time = (time.time() - start_time)
            
            
            # Save both the model and training info in a single file
            save_path = os.path.join(
                self.checkpoint_saving_dir, # print(f"Now: {datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}")
                f"distilled_model_comb_temp_infsteps_{config['num_diffusion_iters']}_steps_{self.model_name_postfix}.pth"
            )
            print(f"Saving model to {save_path}\n"
                  f"  Training time: {end_time:.3f}\n"
                  f"  Current mean loss: {currentDistillStep_mean_loss:.5f}")
            torch.save({
                'model_state_dict': current_nets.state_dict(),
                'training_info': {
                    'training_time': end_time,
                    'current_mean_loss': currentDistillStep_mean_loss,
                    'current_inf_steps': config['num_diffusion_iters'],
                }
            }, save_path)
        
        end_time = (time.time() - start_time)
        print(f"Distillation time: {end_time:.3f} s")

class StudentNetwork_half_steps(ConditionalUnet1D):
    def __init__(self, teacher_network, action_dim, obs_dim, obs_horizon):
        super().__init__(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )
        self.teacher = teacher_network.to(next(self.parameters()).device)
        self.teacher.eval()

    def forward(self, noisy_actions, timesteps, global_cond):
        # Student-Vorhersage
        student_pred = super().forward(
            noisy_actions, timesteps // 2, global_cond
        )
        
        # Lehrer-Vorhersage
        with torch.no_grad():
            teacher_pred = self.teacher(
                noisy_actions, timesteps, global_cond
            )
            if isinstance(teacher_pred, tuple):
                teacher_pred = teacher_pred[0]
        
        return student_pred, teacher_pred




# Progressive Distillation mit config starten
destillation_config = [
            {
                'num_diffusion_iters': 100,  # Original steps
                'temperature': 2.0,          # Soft knowledge transfer
                'epochs': 50,
                'lr': 1e-4
            },
            {
                'num_diffusion_iters': 50,   # Reduced steps
                'temperature': 1.5,          # Moderate knowledge transfer  1.5
                'epochs': 40,
                'lr': 8e-5
            }
        ]

destillation_config_2= [
            {
                'num_diffusion_iters': 100,  # Original steps
                'temperature': 2.0,          # Soft knowledge transfer
                'epochs': 100,
                'lr': 1e-4
            },
            {
                'num_diffusion_iters': 50,   # Reduced steps
                'temperature': 1.0,          # Moderate knowledge transfer  1.0
                'epochs': 80,
                'lr': 8e-5
            }
        ]


'''
distiller = ProgressiveDistillerConfig(
    original_nets=load_model_with_info(
        model_path="checkpoint_dir/ema_nets_model.pth", 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lowdim_obs_dim=3,
        vision_feature_dim=512,
        obs_horizon=2,
        action_dim=7,
        useNoisePred=True), 
    dataloader=create_dataloader(),
    checkpoint_saving_dir=os.path.join(os.getcwd(), 'checkpoint_dir'),
    dest_config=destillation_config_2,
    model_name_postfix="destillation_config_2" # model name: checkpoint_dir/f"distilled_model_comb_temp_infsteps_{config['num_diffusion_iters']}_steps_{self.model_name_postfix}.pth"
).distill()
'''





