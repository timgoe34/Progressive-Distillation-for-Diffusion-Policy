import torch
import zarr
import numpy as np
import torch
import os

#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)


def custom_collate(batch): # ensure that batches have equal sizes of sequence length
    # Separate different keys
    image = [item['image'] for item in batch]
    agent_pos = [item['agent_pos'] for item in batch]
    action = [item['action'] for item in batch]

    #print(f'image: {image}')
    #print(f'agent_pos: {agent_pos}')
    #print(f'action: {action}')    
    
    # Print shapes for debugging
    #print("Shapes before padding:")
    #print("Image shapes:", [x.shape for x in image])
    #print("Agent pos shapes:", [x.shape for x in agent_pos])
    #print("Action shapes:", [x.shape for x in action])
    
    # Pad sequences to the maximum length in this batch
    max_len = max(max(x.shape[0] for x in agent_pos), max(x.shape[0] for x in action))
    
    #print(f"Max length: {max_len}")
    
    def safe_pad(x, max_len):
        if x.shape[0] > max_len:
            return x[:max_len]  # Truncate if longer than max_len
        elif x.shape[0] < max_len:
            pad_width = ((0, max_len - x.shape[0]), (0, 0))
            return np.pad(x, pad_width, mode='constant')
        else:
            return x
    
    agent_pos_padded = [safe_pad(x, max_len) for x in agent_pos]
    action_padded = [safe_pad(x, max_len) for x in action]
    
    # Print shapes after padding for verification
    #print("Shapes after padding:")
    #print("Agent pos shapes:", [x.shape for x in agent_pos_padded])
    #print("Action shapes:", [x.shape for x in action_padded])
    
    # Convert to torch tensors
    image = torch.stack([torch.from_numpy(x) for x in image])
    agent_pos = torch.stack([torch.from_numpy(x) for x in agent_pos_padded])
    action = torch.stack([torch.from_numpy(x) for x in action_padded])
    
    return {
        'image': image,
        'agent_pos': agent_pos,
        'action': action
    }

def create_dataloader(
        dataset_path = "data_storage.zarr",
        pred_horizon = 16,
        obs_horizon = 2,
        action_horizon = 8,
        agent_pos_cutoff_position = 3,
        useSegmented = False # use segmented image data
    ):
    # get demonstration data 
    
    if not os.path.exists(os.path.join(os.getcwd(), dataset_path)):
        print("data_storage.zarr not available!")
        return 0

    # parameters
    
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = ImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        useSegmented=useSegmented, # use segmented image data
        agent_pos_cutoff_position=agent_pos_cutoff_position # if the agent pos looks like this [1,2,3,4,5,6,7] , if this is "3" then it will only take [1,2,3] from the agent pos vector to learn
    )
    
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=False, # windows :(
        collate_fn=custom_collate
    ) 
    return dataloader

def create_sample_indices(episode_ends:np.ndarray, sequence_length:int, pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if idx > 0 and sample_end_idx - sample_start_idx == sequence_length and buffer_end_idx - buffer_start_idx == sequence_length:
                indices.append([
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    #print(f'episode_ends: {episode_ends}')
    #print(f'indices: {indices}')
    #print(f'indices.shape: {indices.shape}')
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    # Add a small epsilon to prevent division by zero
    stats['range'] = np.maximum(stats['max'] - stats['min'], 1e-8)
    return stats

def normalize_data(data, stats):
    # Normalize to [0,1]
    ndata = (data - stats['min']) / stats['range']
    # Clip to ensure we're in the [0,1] range (handles numerical precision issues)
    ndata = np.clip(ndata, 0, 1)
    # Normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    # Unnormalize from [-1, 1] to [0, 1]
    ndata = (ndata + 1) / 2
    # Clip to ensure we're in the [0,1] range (handles numerical precision issues)
    ndata = np.clip(ndata, 0, 1)
    # Unnormalize from [0,1] to original range
    data = ndata * stats['range'] + stats['min']
    return data

# dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 agent_pos_cutoff_position: int, # if the agent pos looks like this [1,2,3,4,5,6,7] , if this is "3" then it will only take [1,2,3] from the agent pos vector to learn
                 useSegmented: bool):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['seg_img'][:] if useSegmented else dataset_root['data']['img'][:]     
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)
        #print(f"train_image_data: {train_image_data}")
        #print(f"train_image_data.shape: {train_image_data.shape}") # (273, 3, 96, 96) passt

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:agent_pos_cutoff_position], # 
            'action': dataset_root['data']['action'][:]
        }

        #print(f"dataset_root['data']['state'][:,:agent_pos_cutoff_position]: {dataset_root['data']['state'][:,:agent_pos_cutoff_position]}")
        #print(f"dataset_root['data']['state'][:,:agent_pos_cutoff_position].shape: {dataset_root['data']['state'][:,:agent_pos_cutoff_position].shape}")
        #print(f"dataset_root['data']['action'][:].shape: {dataset_root['data']['action'][:].shape}")
        #print(f"dataset_root['data']['action'][:]: {dataset_root['data']['action'][:]}")
        episode_ends = dataset_root['meta']['episode_ends'][:]
        #print(f"episode_ends.shape: {episode_ends.shape}")
        #print(f"episode_ends: {episode_ends}")

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        #print("normalized_train_data['agent_pos']: ", normalized_train_data['agent_pos'])
        #print("normalized_train_data['agent_pos']: ", normalized_train_data['action'])
        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        
        #print(f"buffer_start_idx: {buffer_start_idx}")
        #print(f"buffer_end_idx: {buffer_end_idx}")
        #print(f"sample_start_idx: {sample_start_idx}")
        #print(f"sample_end_idx: {sample_end_idx}")

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        #print(f"Sample {idx}: agent_pos: {nsample['agent_pos']}")
        # Check if the sequence has the expected length
        #if nsample['agent_pos'].shape[0] != self.pred_horizon:
        #    return self.__getitem__((idx + 1) % len(self))  # Try the next index
        return nsample