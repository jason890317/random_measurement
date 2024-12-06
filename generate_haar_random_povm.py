import os
import sys
import json
import numpy as np
from povm_generate_function import generate_povm_general
from blended_measurement import blended_measurement
from circuit import compute_full_rank_unitary

def check_blended_measurement(povm, generation_args):
    """
    Check if a blended measurement set is a valid POVM.
    """
    blended_set = blended_measurement(povm, generation_args[1], generation_args[2])
    return is_valid_povm(blended_set)

def check_random_measurement(povm, generation_args):
    """
    Check if a random measurement set is valid for unitary transformations.
    """
    d = generation_args[1]  # System dimension
    return all(is_valid_povm([item, np.eye(d) - item]) for item in povm)

def is_valid_povm(povm, atol=1e-10):
    """
    Verify if a POVM satisfies unitarity conditions.
    """
    unitary = compute_full_rank_unitary(povm)
    # Check if U * U† = I
    return np.allclose(unitary.T.conj() @ unitary, np.eye(unitary.shape[0]), atol=atol)

def ensure_directory_exists(directory):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(directory, exist_ok=True)

def generate_and_save_povm(file_path, generation_function, generation_args, average_time):
    """
    Generate and save POVM sets to a file.

    Parameters:
    - file_path: Path to save the POVM set.
    - generation_function: Function to generate POVM sets.
    - generation_args: Arguments for the POVM generation function.
    - average_time: Number of valid POVM sets to generate.
    """
    try:
        # Load existing POVM set if available
        povm_set = np.load(file_path, allow_pickle=True)
        povm_set = povm_set.tolist()
        if len(povm_set) == average_time or len(povm_set) > average_time:
            print(f"POVM data already exists at {file_path}, no action taken.")
            return
    except FileNotFoundError:
        povm_set = []
    
    while len(povm_set) < average_time:
        
        povm = generation_function(*generation_args)
        
        # Check if the generated POVM is valid
        if check_random_measurement(povm, generation_args) and check_blended_measurement(povm, generation_args):
            povm_set.append(povm)
        else:
            print("Invalid POVM set generated.")
        print(f"\nNumber of valid sets: {len(povm_set)}\n")

    # Save the generated POVM set
    np.save(file_path, povm_set)
    print(f"Generated and saved POVM data to {file_path}")

if __name__ == "__main__":
    
    # Load configuration from JSON file
    with open(f"./test_script/{sys.argv[1]}", 'r') as file:
        data = json.load(file)

    average_time = data["average_time"]  # Number of POVM sets to generate
    case_1_high = data["case_1_high"]  # High probability threshold for case 1
    case_1_low = data["case_1_low"]  # Low probability threshold for case 1
    case_2_pro = data["case_2_pro"]  # Probability threshold for case 2

    ensure_directory_exists('./Haar_measurement_dir')  # Ensure output directory exists

    # Process each test configuration in the JSON data
    for test_data in data["test_data"]:
        state = np.array([np.hstack((1, np.zeros(test_data["dimension"] - 1)))])  # Initialize quantum state
        rho = np.outer(state, state.T.conj())  # Compute density matrix

        method = test_data["method"]
        case = test_data["case"]
        d = test_data["dimension"]
        m = test_data["m"]
        rank = test_data["rank"]

        # Determine case string for file naming
        file_path = f'./Haar_measurement_dir/case_{case}_d_{d}_m_{m}_r_{rank}.npy'  # File path for POVM set                                            

        # Set generation arguments based on case and method
        generation_args=(case,d,m,rank,case_1_high,case_1_low,case_2_pro,rho)
        

        # Generate and save the POVM set
        generate_and_save_povm(file_path, generate_povm_general, generation_args, average_time)
