
from collections import Counter
import numpy as np
import random
import sys



def generate_random_statevector(d):
    # Generate a random complex vector
    vec = np.random.rand(d) + 1j * np.random.rand(d)
    # Normalize the vector
    statevector = vec / np.linalg.norm(vec)
    return statevector  

def show_probability_povm(povm,roh_0,print_pro=False):
    pro=[]  
    for item in povm:
        pro.append(np.trace(item@roh_0))
    if print_pro:
        for item in pro:
            print(abs(item))
    print(np.mean(pro))
    return pro
        

def resolve_blended_result_case_2(counts,m):
   
    n= int(np.log2(m))+1
    accept_time=0
    
    for key,val in counts.items():
        # print("origin: "+key)
        raw_result=key
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(m)]
        result=[ int(item, 2) for item in result]
        # print(result)
        number_counts = Counter(result)
        labels, values = zip(*number_counts.items())
        
        if len(labels)>1:
            accept_time+=val

    return accept_time

def resolve_blended_result_case_1(counts,m):

    accept_time=0
    n= int(np.log2(m))+1
    for key,val in counts.items():
        # print("origin: "+key)
        raw_result=key
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(m)]
        # print(result)
        result=[ int(item, 2) for item in result]
        # print(result)
        result=result[::-1] 
        # print(result)
        for item in result:
            if item != 0 and item == m:
                accept_time+=val
                break
            elif item != 0 and item !=m:
                break
        
    return accept_time

def resolve_random_result_case_1(counts,high):
    
    check=False
    
    for item in high:
        keys=list(counts.keys())
        keys[0]=keys[0][::-1]
        # print(item)
        if(keys[0][item]=='0'):
            check=True
        else:
            check=False
        for j in range(0,item):
            if keys[0][j]=='0':
                check=False

    return check
def xor_binary_strings(str1, str2):
    # Ensure both strings are the same length
    if len(str1) != len(str2):
        raise ValueError("Both strings must be of the same length")

    # Perform XOR on each pair of characters from the two strings
    result = ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(str1, str2))

    return result
def resolve_random_result_case_special(counts,high):
    keys=list(counts.keys())
    
    keys=keys[::-1]
    
    # 0 in the result => accept
    # 1 in the result => reject
    
    ans = ''.join('0' if i in high else '1' for i in range(len(keys[0])))
    # print("ans: "+ans)
    # print(keys)
    ans = xor_binary_strings(ans,keys[0])
    accept_time=0
    for item in ans:
        if item=='0':
            accept_time+=1
        
    return(accept_time) 
   
       
def resolve_blended_result_case_special(counts,m,gate_num_times):
    accept_time=0
    n= int(np.log2(m))+1
    for key,val in counts.items():
        # print("origin: "+key)
        raw_result=key
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(int(gate_num_times*m))]
        # print(result)
        result=[ int(item, 2) for item in result]
        # print(result)
        result=result[::-1] 
        
        print(result)
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        print(correct)
        for i in range(len(result)):
            if result[i]!=0:
                check_array[result[i]-1]=1
                    
        for i in range(len(check_array)):
            if check_array[i]==-1:
                check_array[i]=0
                
        print(check_array)
        xor_result = [a ^ b for a, b in zip(check_array, correct)]
        accept_time=xor_result.count(0)
        
    return accept_time

def resolve_blended_result_case_interweave(counts,m,get_num_times):
    accept_time=0
    n= int(np.log2(m))+1
    for key,val in counts.items():
        # print("origin: "+key)
        raw_result=key
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(int(get_num_times*m))]
        # print(result)
        result=[ int(item, 2) for item in result]
        # print(result)
        result=result[::-1] 
        
        print(result)
        
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        print(correct)
        for i in range(len(result)):
            if result[i]!=0:
                if i%2==0 and check_array[result[i]-1]==-1:
                    check_array[result[i]-1]=0
                elif check_array[result[i]-1]==-1:
                    check_array[result[i]-1]=1
        for i in range(len(check_array)):
            if(check_array[i]==-1):
                check_array[i]=random.choice([0,1])
        print(check_array)
        xor_result = [a ^ b for a, b in zip(check_array, correct)]
        accept_time=xor_result.count(0)
    return accept_time


def resolve_blended_three_result(counts,m,permutation,table,round,special=True):
    keys=list(counts.keys())[0]
                    
    # print(keys)
    result=[ int(keys[2*i:(i+1)*2],2) for i in range(round)]
    result=result[::-1]
    
    # print("result: "+ str(result))
    for i in range(len(result)):
        if result[i]==1:
            for j in range(len(permutation[i])):
                if permutation[i][j]==1:
                    table[j]+=1
        elif result[i]==2:
            for j in range(len(permutation[i])):
                if permutation[i][j]==2:
                    table[j]+=1
    if not special:
        # print(table)
        max_index = np.argmax(table)
        # print(max_index)
        number_counts = Counter(table)
        # print(number_counts[table[max_index]])
        if max_index==m-1 and number_counts[table[max_index]]==1:
            accept+=1
            # print(table)
        
    else:
        # print(table)
        
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        
        for item in top_half_indices(table):
            check_array[item-1]=1
        for i in range(len(check_array)):
            if check_array[i]==-1:
                check_array[i]=0
        xor_result = [a ^ b for a, b in zip(check_array, correct)]
        accept=xor_result.count(0)
        
    return accept


def print_progress(current, total, bar_length=20):
    
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = (bar_length - len(arrow)) * ' '
    progress_bar = f'\r{current}/{total}: [{arrow}{padding}]'
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    
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

def top_half_indices(array):
    # Calculate the number of elements to select
    num_elements = len(array) // 2

    # Get unique values and their indices
    unique, indices, counts = np.unique(array, return_inverse=True, return_counts=True)

    # Create a list to store the selected indices
    selected_indices = []

    # Iterate over the unique values in descending order of their value
    for value in np.flip(unique):
        # Find the indices of the current value
        current_indices = np.where(array == value)[0]
        
        # If adding all current_indices exceeds num_elements, randomly sample from them
        if len(selected_indices) + len(current_indices) > num_elements:
            needed = num_elements - len(selected_indices)
            selected_indices.extend(np.random.choice(current_indices, needed, replace=False))
            break
        else:
            selected_indices.extend(current_indices)

    return selected_indices

def generate_permutations(m, num):
    if m % 2 != 0:
        raise ValueError("m must be an even number.")

    sample_size = num
    
    sampled_permutations=[]
    base_array = [1] * (m // 2) + [2] * (m // 2)
    while len(sampled_permutations)!=sample_size:
        permuted_array = np.random.permutation(base_array)
        sampled_permutations.append(permuted_array)
    
    # print(sampled_permutations)
    return sampled_permutations