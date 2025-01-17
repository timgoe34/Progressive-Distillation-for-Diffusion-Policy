"""
This python file does:

- defines some analytical functions to 
    - analyze inference data collected during evaluation of a model

"""
# IMPORTS
import os
import numpy as np

def run():
    

    def analyze_inference_data(file_path):
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
        if not all(field in inference_array[0].dtype.names for field in ['score', 'inf_time']):
            print("The file structure is not compatible.")
            return

        
        # Calculate average epoch time and trajectory length
        avg_epoch_time = np.mean([record['inf_time'] for record in inference_array])
        avg_score = np.mean([record['score'] for record in inference_array])

        print(f"Average Epoch Time: {avg_epoch_time:.4f} seconds")
        print(f"Average Score: {avg_score:.5f} seconds")

    
    analyze_inference_data(r'infData\ema_student_1_stepspush_t.pth_infData_runs_1000_steps_1.npy')
    analyze_inference_data(r'infData\ema_student_3_stepspush_t.pth_infData_runs_1000_steps_2.npy')
   
    
if __name__ == "__main__":
    run()