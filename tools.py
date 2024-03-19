from itertools import product
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def generate_random_projector(d):
    # Generate a random complex vector
    vec = np.random.rand(d) + 1j * np.random.rand(d)
    
    # Normalize the vector
    vec_normalized = vec / np.linalg.norm(vec)
    
    # Construct the projector
    projector = np.outer(vec_normalized, np.conj(vec_normalized))
    
    return projector


def generate_random_statevector(d):
    # Generate a random complex vector
    vec = np.random.rand(d) + 1j * np.random.rand(d)
    # Normalize the vector
    statevector = vec / np.linalg.norm(vec)
    return statevector  

def print_eigenvalue(povm):
    for item in povm:
        eigenval,_=np.linalg.eigh(item)
        print(eigenval)

def show_probability_povm(povm,roh_0,print_pro=False):
    pro=[]
    for item in povm:
        pro.append(np.trace(item@roh_0))
    if print_pro:
        for item in pro:
            print(item)
    return pro
        


def generate_binary_strings(n):
    # Generate all combinations of 0 and 1 of length n
    combinations = product('01', repeat=n)
    
    # Join each combination into a string and create a set of these strings
    binary_strings = {''.join(combination) for combination in combinations}
    
    return binary_strings


def resolve_blended_result_case_2(counts,m):
    for key in counts.keys():
        raw_result=key
    n= int(np.log2(m))+1
    result=[raw_result[n*i:(i+1)*n] for i in range(m)]
    result=[ int(item, 2) for item in result]
    number_counts = Counter(result)

    return number_counts

def resolve_blended_result_case_1(counts,m):
    # print(counts)
    for key in counts.keys():
        raw_result=key
    n= int(np.log2(m))+1
    result=[raw_result[n*i:(i+1)*n] for i in range(m)]
    # print(result)
    result=[ int(item, 2) for item in result]
    
    for item in result:
        if item != 0 and item == m:
            
            return True
        elif item != 0 and item !=m:
            
            return False
        
        else:
            return False
    
    


def plot_sequential_blended_result(labels,values,m):
    plt.figure(figsize=(20, 6))  # Optional: Adjust the figure size
    plt.bar(labels, values, color='skyblue')  # Create a bar chart

    # Add title and labels to the plot
    plt.title('Frequency of Each E_i')
    plt.xlabel('total number of the Ei :' + str(m+1))
    plt.ylabel('measurement times : '+ str(m))

    # Show the plot
    plt.xticks(range(0, m+1))  # Ensure all numbers are shown on x-axis
    plt.show()

def print_progress(current, total, bar_length=20):
    
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = (bar_length - len(arrow)) * ' '
    progress_bar = f'\r{current}/{total}: [{arrow}{padding}]'
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    

def generate_rank_n_projector(rank, dim):
    """
    Generate an arbitrary rank projector in a given dimension.
    
    Parameters:
    - dim: The dimension of the space.
    - rank: The rank of the projector (must be <= dim).
    
    Returns:
    - A numpy array representing the projector matrix.
    """
    if rank > dim:
        raise ValueError("Rank cannot be greater than dimension.")
    
    # Step 1: Generate random vectors
    np.random.seed(int(time.time()))
    random_vectors = np.random.rand(rank, dim)
    
    # Step 2: Orthogonalize the vectors using QR decomposition
    q, _ = np.linalg.qr(random_vectors.T)  # Transpose to get dim x rank matrix
    
    # Step 3: Construct the projector by summing outer products of orthonormal vectors
    projector = np.zeros((dim, dim))
    for i in range(rank):
        projector += np.outer(q[:, i], q[:, i].conj())
    
    eig,vec=np.linalg.eig(projector)
    
    return projector