
from generate_povm import generate_povm_epson_case_1,generate_povm_epson_case_2,generate_povm_epson_case_special
import numpy as np
import json
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_and_save_povm(file_path, generation_function, generation_args):
    """Generate and save POVM data if it doesn't already exist."""
    try:
        existed_set=np.load(file_path)
    except :
        povm_set = [generation_function(*generation_args) for _ in range(average_time)]
        np.save(file_path, povm_set)
        print(f"Generated and saved POVM data to {file_path}")
    else:
        if  len(existed_set)!=average_time:
            povm_set = [generation_function(*generation_args) for _ in range(average_time)]
            np.save(file_path, povm_set)
            print(f"Generated and saved POVM data to {file_path}")
        else:
            print(f"POVM data already exists at {file_path}, no action taken.")
    
            
if __name__ == "__main__":
    # Load configuration from JSON file
    with open('test_data.json', 'r') as file:
        data = json.load(file)

    # Extract common parameters
    average_time = data["average_time"]
    case_1_high = data["case_1_high"]
    case_1_low = data["case_1_low"]
    case_2_pro = data["case_2_pro"]
    method_s=data["method_s"]
    average_time=data["average_time"]
    m_s=data["m_s"]
    d_s=data["d_s"]
    rank_s=data["rank_s"]

    # Ensure the measurement directory exists
    measurement_dir = './measurement_dir'
    ensure_directory_exists(measurement_dir)

    # Process each test data entry
    for test_data in data["test_data"]:
        state = np.array([np.hstack((1, np.zeros(test_data["d"] - 1)))])
        roh_0 = np.outer(state, state.T.conj())

        method = test_data["method"]
        case = test_data["case"]
        d = test_data["d"]
        m = test_data["m"]
        rank = test_data["rank"]

        # File naming convention maintained as per the original method and case description
        if method in ["blended", "random"]:
            case_str = f"case_{case}"  # Case description (e.g., 'case_1')
        else:
            case_str = "special"  # For special methods, use a general 'special' tag

        # Construct the file path using the detailed method, case, dimensions, measurements, and rank
        file_path = f'{measurement_dir}/{case_str}_d_{d}_m_{m}_r_{rank}.npy'

        # Select the appropriate POVM generation function based on the method and case
        if method in ["blended", "random"]:
            generation_function = generate_povm_epson_case_1 if case == 1 else generate_povm_epson_case_2
            generation_args=(d, m, rank, case_1_high, case_1_low, roh_0) if case == 1 else (d,m,rank,case_2_pro,roh_0)
        elif method in ['special_random', 'special_blended', 'interweave', 'blended_three','classical_shadow','optimizing_blended']:
            generation_function = generate_povm_epson_case_special
            generation_args=(d, m, rank, case_1_high, case_1_low, roh_0)
        else:
            continue  # Skip if the method is not recognized

        # Generate and save the POVM data if it doesn't exist
        generate_and_save_povm(file_path, generation_function, generation_args)
