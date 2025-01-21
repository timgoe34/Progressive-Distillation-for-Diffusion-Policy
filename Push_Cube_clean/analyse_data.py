"""
This python file does:

- defines some analytical functions to 
    - rank the quality of training data
    - plot trajectories
    - analyze inference data collected during evaluation of a model

"""
# IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import zarr
from scipy.stats import variation
from sklearn.neighbors import KernelDensity
import pandas as pd

def run():

    ''' Visualize Trajectories'''


    def plot_trajectories(dataset_path, traj_number=-1):
        """
        Plot multiple trajectories with different colors.
        
        Args:
            dataset_path: path to zarr dataset
            traj_number: if only one trajectory should be displayed, number of this trajectory

        How to use: 
            plot_trajectories(dataset_path) (all trajectories)
            or    
            plot_trajectories(dataset_path, traj_number= 86) (trajectory 86)
        """
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        positions = dataset_root['data']['state'][:,:3]
        episode_ends = dataset_root['meta']['episode_ends'][:]
        print(f"Number of trajectories: {len(episode_ends)}")
        
        # Create a new 3D figure with more space for labels
        plt.rcParams.update({'font.size': 14})  # Increase default font size
        fig = plt.figure(figsize=(14, 10))  # Larger figure to accommodate bigger labels
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate episode start indices
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        
        # Generate colors using HSV colormap for better distinction
        n_trajectories = len(episode_ends)
        colors = plt.cm.hsv(np.linspace(0, 1, n_trajectories))
        
        if traj_number >= 0 and traj_number < len(episode_ends):

            trajectory = positions[episode_starts[traj_number]:episode_ends[traj_number]]
            #print(f"trajectory[0]: {trajectory[0]}")
            ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    '-o', markersize=2, label=f'Trajectory {traj_number}',
                    color=colors[traj_number], alpha=0.7)
            ax.set_title(f'Robot End-Effector Trajectory (3D) - Trajectory: {traj_number}', fontsize=18, pad=20)
        else:
            # Plot each trajectory
            for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
                
                trajectory = positions[start:end]
                #print(f"trajectory[0]: {trajectory[0]}")
                ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        '-o', markersize=2, label=f'Trajectory {i+1}',
                        color=colors[i], alpha=0.7)
            ax.set_title('Robot End-Effector Trajectories (3D)', fontsize=18, pad=20)
        
        # Set labels with larger font sizes
        ax.set_xlabel('X Position', fontsize=16, labelpad=15)
        ax.set_ylabel('Y Position', fontsize=16, labelpad=15)
        ax.set_zlabel('Z Position', fontsize=16, labelpad=15)
        
        
        # Increase tick label sizes
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)
        
        ax.grid(True, alpha=0.3)
        
        # Add legend with larger font
        if n_trajectories > 10:
            ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), 
                    ncol=1, title='First 10 trajectories',
                    fontsize=12, title_fontsize=14)
            for i in range(10, n_trajectories):
                ax.plot([], [], [], '-', color=colors[i], alpha=0.7)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), 
                    ncol=1, fontsize=12)
        
        # Add some default viewing angles
        ax.view_init(elev=20, azim=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        print("Tipp: Du kannst die 3D-Ansicht mit der Maus drehen!")
        
        plt.show()




    def analyze_trajectories(dataset_path):
        """
        Analyze trajectories for smoothness based on point distances and overshooting.
        
        Args:
            dataset_path: path to zarr dataset relative to this script
        
        Returns:
            DataFrame with smoothness metrics for each trajectory

        How to use: 
            results_df = analyze_trajectories(dataset_path)
        """
        dataset_path = os.path.join(os.getcwd(), dataset_path)

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        positions = dataset_root['data']['state'][:,:3]
        episode_ends = dataset_root['meta']['episode_ends'][:]
            
        # Calculate episode start indices
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        
        # Define regions
        start_region = {
            'min': np.array([0.28, -0.35]),
            'max': np.array([0.4, 0.35])
        }
        target_point = np.array([0.5, 0.0]) 
        radius = 0.02 # radius around goal point that counts as done
        belt_width = 0.1 # belt thickness around start_region and target_point

        covered_threshold = 0.1 # determines how finely the start region is checked for coverage
        
        results = []
        all_trajectories = []  # Store complete trajectories for diversity analysis
        
        for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            trajectory = positions[start:end]
            all_trajectories.append(trajectory)
            
            metrics = {}
            metrics['trajectory_id'] = i + 1
            
            # 1. Smoothness Analysis (as before)
            distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
            metrics['distance_cv'] = variation(distances)
            
            vectors = np.diff(trajectory, axis=0)
            vectors_normalized = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
            dot_products = np.sum(vectors_normalized[:-1] * vectors_normalized[1:], axis=1)
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
            sharp_turns = np.sum(angles > np.pi/2)
            metrics['sharp_turns_ratio'] = sharp_turns / len(angles) if len(angles) > 0 else 0
            
            # 2. Error Handling Analysis - Belt Detection
            # Count points in start region belt
            start_belt_points = sum(is_in_belt(p, 
                                            start_region['min'], 
                                            start_region['max'], 
                                            belt_width) for p in trajectory)
            
            # Count points in target region belt
            target_belt_points = sum(radius < np.linalg.norm(p[:2] - target_point) < belt_width for p in trajectory)
            
            metrics['start_belt_points'] = start_belt_points
            metrics['target_belt_points'] = target_belt_points
            
            # Calculate scores
            metrics['smoothness_score'] = 100 * (1 - np.clip(0.5 * metrics['distance_cv'] + 
                                                            0.5 * metrics['sharp_turns_ratio'], 0, 1))
            metrics['error_handling_score'] = 100 * (
                0.5 * min(start_belt_points / 20, 1) +  # Normalize to max 20 points
                0.5 * min(target_belt_points / 20, 1)
            )
            
            results.append(metrics)
        
        # 3. Coverage Analysis
        # Create grid over start region
        x_grid = np.linspace(start_region['min'][0], start_region['max'][0], 20)
        y_grid = np.linspace(start_region['min'][1], start_region['max'][1], 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Calculate coverage using KDE
        all_points = np.vstack([t[:, :2] for t in all_trajectories])
        kde = KernelDensity(bandwidth=0.05)
        kde.fit(all_points)
        coverage_scores = np.exp(kde.score_samples(grid_points))
        coverage_score = 100 * (np.mean(coverage_scores > covered_threshold))  # Threshold for "covered"
        
        # 4. Trajectory Variance Analysis
        # Compare each trajectory with every other trajectory
        n_trajectories = len(all_trajectories)
        variance_scores = []
        
        for i in range(n_trajectories):
            for j in range(i+1, n_trajectories):
                # Resample trajectories to same length for comparison
                traj1 = all_trajectories[i]
                traj2 = all_trajectories[j]
                
                # Compute average distance between trajectories
                min_len = min(len(traj1), len(traj2))
                traj1_resampled = traj1[:min_len]
                traj2_resampled = traj2[:min_len]
                
                mean_distance = np.mean(np.linalg.norm(traj1_resampled - traj2_resampled, axis=1))
                variance_scores.append(mean_distance)
        
        trajectory_diversity = np.mean(variance_scores) if variance_scores else 0
        diversity_score = 100 * min(trajectory_diversity / 0.2, 1)  # Normalize to expected maximum
        
        # Add global scores to DataFrame
        df_results = pd.DataFrame(results)
        df_results['coverage_score'] = coverage_score
        df_results['diversity_score'] = diversity_score
        
        # Plotting
        fig = plt.figure(figsize=(20, 10))
        
        # 1. All trajectories with belt regions
        ax1 = fig.add_subplot(221)
        for traj in all_trajectories:
            ax1.plot(traj[:, 0], traj[:, 1], alpha=0.3)
        
        # Plot start region and belt
        rect = plt.Rectangle(start_region['min'], 
                            start_region['max'][0] - start_region['min'][0],
                            start_region['max'][1] - start_region['min'][1],
                            fill=False, color='red')
        ax1.add_patch(rect)
        rect_outer = plt.Rectangle(start_region['min'] - belt_width,
                                start_region['max'][0] - start_region['min'][0] + 2*belt_width,
                                start_region['max'][1] - start_region['min'][1] + 2*belt_width,
                                fill=False, color='blue')
        ax1.add_patch(rect_outer)
        ax1.set_title('Trajektorien und Regionen', fontsize=16,)
        
        # 2. Coverage density
        ax2 = fig.add_subplot(222)
        coverage_plot = ax2.tricontourf(grid_points[:, 0], grid_points[:, 1], 
                                    coverage_scores, levels=20)
        plt.colorbar(coverage_plot, ax=ax2)
        ax2.set_title('Abdeckungsdichte', fontsize=16)
        
        # 3. Trajectory diversity visualization
        ax3 = fig.add_subplot(223)
        plt.hist(variance_scores, bins=20)
        ax3.set_title('Trajektorien Diversitäts-Verteilung', fontsize=16)
        
        # 4. Scores
        ax4 = fig.add_subplot(224)
        scores = [df_results['smoothness_score'].mean(),
                df_results['error_handling_score'].mean(),
                coverage_score,
                diversity_score]
        plt.bar(['Smoothness Glätte', 'Fehlerhandling', 'Abdeckung', 'Diversität'], scores)
        ax4.set_ylim(0, 100)
        ax4.set_title('Durchschnittliche Bewertungen', fontsize=16)
        
        # Print summary
        print("\nTrajectory Analysis Summary:")
        print("-" * 50)
        print(f"Number of trajectories: {len(df_results)}")
        print("\nAverage Scores:")
        print(f"Smoothness: {df_results['smoothness_score'].mean():.2f}")
        print(f"Error Handling: {df_results['error_handling_score'].mean():.2f}")
        print(f"Coverage: {coverage_score:.2f}")
        print(f"Diversity: {diversity_score:.2f}")
        
        plt.tight_layout()
        plt.show()
        
        return df_results
        
    def is_in_belt(point, region_min, region_max, belt_width=0.02):
        """Check if a point is in the belt around a rectangular region"""
        x, y = point[:2]
        x_min, y_min = region_min
        x_max, y_max = region_max
        
        # Inner and outer boundaries
        x_inner_min = x_min - belt_width
        x_inner_max = x_max + belt_width
        y_inner_min = y_min - belt_width
        y_inner_max = y_max + belt_width
        
        # Check if point is in belt
        in_outer = (x_inner_min <= x <= x_inner_max and 
                    y_inner_min <= y <= y_inner_max)
        in_inner = (x_min <= x <= x_max and 
                    y_min <= y <= y_max)
        return in_outer and not in_inner 

    #results_df = analyze_trajectories(dataset_path)

    def analyze_inference_data(file_path, ax, titel):
        file_path = os.path.join(os.getcwd(), file_path)
        print(f"os.getcwd(): {os.getcwd()}")
        # Check if file exists
        if os.path.exists(file_path):
            print(f"{file_path} exists.")
        else:
            print(f"{file_path} does not exist.")
            return

        # Load the saved data
        inference_array = np.load(file_path, allow_pickle=True)

        # Check data structure in .npy file
        if not all(field in inference_array[0].dtype.names for field in ['epoch_time', 'traj_length', 'trajectory', 'status', 'cube_pos']):
            print("The file structure is not compatible.")
            return

        # Limit to the first 1000 rows if there are more than 1000
        if len(inference_array) > 1000:
            inference_array = inference_array[:1000]
        # Calculate average epoch time and trajectory length
        avg_epoch_time = np.mean([record['epoch_time'] for record in inference_array])
        avg_traj_length = np.mean([record['traj_length'] for record in inference_array])
        avg_inf_time = inference_array['inf_time_avg'][len(inference_array)-1]

        # Count successful and unsuccessful episodes
        successful_episodes = sum(1 for record in inference_array if record['status'] == 'done')
        unsuccessful_episodes = sum(1 for record in inference_array if record['status'] == 'not done')

        print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")
        print(f"Average Inf Time: {avg_inf_time:.5f} seconds")
        print(f"Average Trajectory Length: {avg_traj_length:.0f}")
        print(f"Successful Episodes: {successful_episodes}")
        print(f"Unsuccessful Episodes: {unsuccessful_episodes}")

        added_legend_done = False
        added_legend_not_done = False

        for record in inference_array:
            trajectory = np.array(record['trajectory'])
            status = record['status']
            cube_pos = record['cube_pos']

            # Plot trajectory
            if status == "done":
                ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', alpha=0.2, label="Successful Trajectory" if not added_legend_done else "")
                added_legend_done = True
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color='red', alpha=0.3, label="Unsuccessful Trajectory" if not added_legend_not_done else "")
                added_legend_not_done = True

            # Plot cube position for unsuccessful episodes
            if status == "not done":
                ax.scatter(cube_pos[0], cube_pos[1], color='r', marker='x', label="Cube Position" if not added_legend_not_done else "")

        ax.set_xlabel("X", fontsize=19)
        ax.set_ylabel("Y", fontsize=19)
        ax.set_title(titel, fontsize=19)
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=19)

    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    #fig.suptitle("Inferenz Trajektorien Vergleich", fontsize=18)

    # Analyze each dataset
    
    #analyze_inference_data(r'modell_200_episodes_gut\data_segmented_gut_inf10.npy', ax2, titel="Lösungsbahnen 3 - segmented")
    #analyze_inference_data(r'2 inference steps test results\data_rgb_gut_100epochs_inf2_normal_model.npy', ax1, titel="Lösungsbahnen normales Modell")

    # Destillierte

    #analyze_inference_data(r'2 inference steps test results\data_rgb_gut_100epochs_inf2_distilled_model_no_temp.npy', ax2, titel="Lösungsbahnen destilliert - Model no temp scaling")

    #analyze_inference_data(r'2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps_1049.215_timeInSec_0.013926_loss.npy', ax3, titel="Lösungsbahnen destilliert - config 1")
    """
    # Distillation configuration für '2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps.npy'
    - Modell ist trainiert worden mit Zeitschritthalbierung auf 50 Schritte
    - mit 25 DDPM-Trainings-Schritten und 2 Inf Schritten ausgewertet
        self.distillation_configs = [
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
    """

    #analyze_inference_data(r'2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps_12-12-2024_23-09-02.npy', ax4, titel="Lösungsbahnen destilliert - config 2")
    """
    # Distillation configuration für '2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps.npy'
    - Modell ist trainiert worden mit Zeitschritthalbierung auf 50 Schritte
    - mit 25 DDPM-Trainings-Schritten und 2 Inf Schritten ausgewertet
        self.distillation_configs = [
            {
                'num_diffusion_iters': 100,  # Original steps
                'temperature': 2.0,          # Soft knowledge transfer
                'epochs': 100,
                'lr': 1e-4
            },
            {
                'num_diffusion_iters': 50,   # Reduced steps
                'temperature': 1.0,          # Moderate knowledge transfer  1.5
                'epochs': 100,
                'lr': 8e-5
            }
        ]
    """

    #analyze_inference_data(r'2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps_destillation_config_2_42_seed_ema_nets_model_1.pth.npy', ax2, titel="Lösungsbahnen destilliert - config 2 - seed 42")

    #analyze_inference_data(r'2 inference steps test results\distilled_model_comb_temp_infsteps_50_steps_destillation_config_2_1000_seed_ema_nets_model_2.pth.npy', ax1, titel="Lösungsbahnen destilliert - config 2 - seed 1000")
    # Adjust layout to prevent overlap
    #analyze_inference_data(r'2 inference steps test results\seed_42_ema_student_3_steps_inf_2.pth.npy', ax3, titel="Lösungsbahnen destilliert - dest3")
    analyze_inference_data(r'1 inference steps test results\42_seed_400ep_angles_ema_nets_model.pth.npy', ax1, titel="Lösungsbahnen - 400ep_angles")
    analyze_inference_data(r'1 inference steps test results\ema_student_1_steps400eps.pth.npy', ax2, titel="Lösungsbahnen destilliert 1 Schritt - ema_student_1_steps400eps.pth.npy")
    analyze_inference_data(r'2 inference steps test results\ema_student_3_steps400eps.pth.npy', ax3, titel="Lösungsbahnen destilliert 2 Schritte- ema_student_3_steps400eps.pth.npy")
    analyze_inference_data(r'4 inference steps test results\ema_student_3_steps400eps_inf4.pth.npy', ax4, titel="Lösungsbahnen destilliert 4 Schritte - ema_student_3_steps400eps.pth.npy")
    plt.tight_layout()
    plt.show()
   
    
if __name__ == "__main__":
    run()