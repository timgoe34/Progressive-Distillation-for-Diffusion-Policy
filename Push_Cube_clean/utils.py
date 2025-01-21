"""
This python file does:

- defines some helper functions like 
    is_valid_filename_practical to detect valid file names
    load_model_with_info to load saved models

"""

# IMPORTS
import torch
import os
import torch
import torch.nn as nn

# import u net functions
from network import *
# import vision encoder functions
from vision_encoder import *

def is_valid_filename_practical(filename):
    # checks if the file name can be saved and if it is valid
    try:
        with open(filename, 'w') as f:
            pass
        os.remove(filename)  # Clean up the test file
        return True
    except OSError:
        return False
    
def load_model_with_info(model_path,            # file_path: Path to the .pth file containing the model and training info.
                         device,                # The device to load the model onto.
                         lowdim_obs_dim,        # low dim obs of the model
                         vision_feature_dim,    # Vision feature dimension
                         obs_horizon,           # Observation horizon
                         action_dim,            # dim of actions
                         useNoisePred           # use noise Prediction Network or Sample Prediction Network?
                         ):
    """
    Load a model and optionally display training information.
            
    Returns:
        model: Loaded model.
    """
    # construct ResNet18 encoder
   
    vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))

    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    noise_pred_net, nets = 0,0
    if useNoisePred:
        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        }) 
    else:
       # create sample network object
        sample_pred_net = ConditionalUnet1D_sample(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'sample_pred_net': sample_pred_net
        })   
    # Load model
    if os.path.exists(model_path):
        print(f"model_path exists!")
        #model_checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        
    
        # Check if training info exists
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            print("Training Information:")
            for key, value in training_info.items():
                print(f"  {key}: {value}")

            nets.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nets = nets.to(device) # map nets to correct device!   
            return nets
        else:
            print("No training information found in checkpoint.")

            print("Loading nets from modelpath...")
            nets.load_state_dict(checkpoint, strict=False)
            nets = nets.to(device) # map nets to correct device!    
                        
            print("Model loaded successfully.")
            
            return nets
    else:
        print("model_path invalid!")
        return 0