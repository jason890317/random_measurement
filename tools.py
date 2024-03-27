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
    # for key in counts.keys():
    #     raw_result=key
    # n= int(np.log2(m))+1
    # result=[raw_result[n*i:(i+1)*n] for i in range(m)]
    # result=[ int(item, 2) for item in result]
    # number_counts = Counter(result)

    # return number_counts
    n= int(np.log2(m))+1
    accept_time=0
    for key,val in counts.items():
        raw_result=key
        result=[raw_result[n*i:(i+1)*n] for i in range(m)]
        result=[ int(item, 2) for item in result]
        number_counts = Counter(result)
        labels, values = zip(*number_counts.items())
        # print(labels)
        if len(labels)>1:
            accept_time+=val

    return accept_time
def resolve_blended_result_case_1(counts,m):
    
    # for key in counts.keys():
    #     raw_result=key
    # n= int(np.log2(m))+1
    # result=[raw_result[n*i:(i+1)*n] for i in range(m)]
    
    # result=[ int(item, 2) for item in result]
    # print(result)
    # for item in result:
    #     if item != 0 and item == m:
            
    #         return True
    #     elif item != 0 and item !=m:
            
    #         return False
        
    #     else:
    #         return False
    accept_time=0
    n= int(np.log2(m))+1
    for key,val in counts.items():
        # print(f'{key}:{val}')
        raw_result=key
        result=[raw_result[n*i:(i+1)*n] for i in range(m)]
        result=[ int(item, 2) for item in result]
        # print("result: "+str(result))
        for item in result:
            if item != 0 and item == m:
                accept_time+=val
                break
            elif item != 0 and item !=m:
                break
        # print("accept_tiem:"+str(accept_time))
    return accept_time


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

def projector_html(dir_name,rotated_pro,top_num):
    
    html_content=""
    with open(dir_name, 'a') as file:
        
        flat_matrix = rotated_pro.flatten()
        magnitudes = np.abs(flat_matrix)
        indices_of_largest = np.argpartition(magnitudes, -top_num)[-top_num:]
        mask = np.zeros_like(magnitudes, dtype=bool)
        mask[indices_of_largest] = True
        modified_flat_matrix = np.where(mask, flat_matrix, 0)
        # print(modified_flat_matrix)
        # Reshaping back to the original matrix shape
        modified_matrix = modified_flat_matrix.reshape(rotated_pro.shape)
    # Iterate through the matrix row by row
        # rotated_pro=np.abs(rotated_pro)
        for row in modified_matrix:
            # Create a string for the row
            # html_content = html_content+' '.join(f"{np.abs(z):.4f}" for z in row)+"<br>"
            for z in row:
                if(z!=0):
                    html_content=html_content+"<span style='color:red;'>"+f"{np.abs(z):.4f} "+"</span>"
                else:
                    html_content=html_content+f"{np.abs(z):.4f} "
            html_content=html_content+"<br>"
        html_content="<p>"+html_content+"</p>"
        file.write(html_content)
