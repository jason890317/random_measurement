
from generate_povm import generate_povm_general
import numpy as np
import json
import os
from scipy.linalg import sqrtm
from scipy.linalg import svd
from blended_measurement import blended_measurement
import sys
from sample import sample_povm_general
def checkBlednedMeasurement(povm,generation_args):
    blended_set=blended_measurement(povm,generation_args[1],generation_args[2])
    # blended_set=[ item@item.T.conj() for item in blended_set]
   
    return isValidityPOVM(blended_set)
def checkRandomMeasurement(povm,generation_args):
    
    d=generation_args[1]
    for item in povm:
        item_inv=np.eye(d)-item
        if not isValidityPOVM([item,item_inv]):
            return False
    return True
        

def isValidityPOVM(povm, atol=1e-10, rtol=0):
    
    povm = [sqrtm(M)for M in povm]
   
    v = np.hstack(povm) # arrange the povm elements to form a matrix of dimension: Row X Col*len(povm)
    v = np.atleast_2d(v) # convert to 2d matrix
    v=v.astype('complex128')
    u, s, vh = svd(v)    # apply svd
    tol = atol
    nnz = (s >= tol).sum()
    ns = vh[nnz:]         # missing rows of v
    V = np.vstack((v, ns)) 
    n = int(np.ceil(np.log2(V.shape[0])))
    N = 2**n      # dimension of system and ancilla Hilber space
    r,c = V.shape  
    U = np.eye(N, dtype=complex) # initialize Unitary matrix to the identity. Ensure it is complex
    U[:r,:c] = V[:r,:c] # assign all the elements of V to the corresponding elements of U
    
    U = U.conj().T  # Transpose the unitary so that the rows are the povm
    
    if np.allclose(U.T.conj()@U, np.eye(N),atol=(1e-10)):
        return True
    else: 
        return False



def ensure_directory_exists(directory):
    """Ensure the specified directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_and_save_povm(file_path, generation_function, generation_args):
    """Generate and save POVM data if it doesn't already exist."""
    try:
        existed_set=np.load(file_path)
    except :
        
        povm_set=[]
        while len(povm_set)!=average_time:
            povm=generation_function(*generation_args)
            if checkRandomMeasurement(povm,generation_args) and checkBlednedMeasurement(povm,generation_args):
                povm_set.append(povm)
                
            else:
                print("failed")
            print("\nnumber of set: "+str(len(povm_set))+"\n")
        np.save(file_path, povm_set)
        print(f"Generated and saved POVM data to {file_path}")
    else:
        if  len(existed_set)!=average_time:
            povm_set=[]
            while len(povm_set)!=average_time:
                povm=generation_function(*generation_args)
                if checkRandomMeasurement(povm,generation_args) and checkBlednedMeasurement(povm,generation_args):
                    povm_set.append(povm)
                    
                else:
                    print("failed")
                print("number of set: "+str(len(povm_set)))
            np.save(file_path, povm_set)
            print(f"Generated and saved POVM data to {file_path}")
        else:
            print(f"POVM data already exists at {file_path}, no action taken.")
    
            
if __name__ == "__main__":
    # Load configuration from JSON file
    
    test_data_path= "./test_script/"+sys.argv[1]
    with open(test_data_path, 'r') as file:
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
    measurement_dir = './Haar_measurement_dir'
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
            generation_function = generate_povm_general
            generation_args=(1,d, m, rank, case_1_high, case_1_low, case_2_pro,roh_0) if case == 1 else (2,d, m, rank, case_1_high, case_1_low, case_2_pro,roh_0)
        elif method in ['special_random', 'special_blended', 'interweave', 'blended_three','classical_shadow','optimizing_blended']:
            generation_function = generate_povm_general
            generation_args=(3,d, m, rank, case_1_high, case_1_low, case_2_pro,roh_0)
        else:
            continue  # Skip if the method is not recognized

        # Generate and save the POVM data if it doesn't exist
        generate_and_save_povm(file_path, generation_function, generation_args)
