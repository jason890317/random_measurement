
from collections import Counter
import numpy as np
import random
import sys


def compute_probability_for_povm_set(povm, rho, print_pro=False):
    pro = [np.trace(item @ rho) for item in povm]
    if print_pro:
        for item in pro:
            print(abs(item))
    return pro
        
    
def check_outcome_condition_for_blended_case_2(counts, m):

    n = int(np.log2(m)) + 1
    keys = list(counts[0].keys())[0]
    raw_result = keys

    result = [int(raw_result[n * i:(i + 1) * n], 2) for i in range(m)]
    result = result[::-1]

    number_counts = Counter(result)

    return len(number_counts) > 1

def check_outcome_condition_for_blended_case_1(counts,m):

    n= int(np.log2(m))+1
    keys=list(counts[0].keys())[0]

    raw_result=keys

    result=[raw_result[n*i:(i+1)*n] for i in range(m)]
  
    result=[ int(item, 2) for item in result][::-1] 

    for item in result:
        if item != 0:
            return item == m
    return False
    

def check_outcome_condition_for_random_case_1(counts,high):
    check = False
    keys = list(counts[0].keys())[0][::-1]
    high_index = high[0][0]

    for i, key in enumerate(keys):
        if key == '0':
            if i == high_index:
                check = True
            else:
                break
    return check

def check_outcome_condition_for_random_case_2(counts):
    
    keys=list(counts[0].keys())
    keys[0]=keys[0]
    for item in keys[0]:
        if item=='0':
            return True
    return False
    
    
def xor_binary_strings(str1, str2):

    if len(str1) != len(str2):
        raise ValueError("Both strings must be of the same length")

    result = ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(str1, str2))
    
    return result


def check_outcome_condition_for_random_case_special(count_set,indices_set,m):
    
    accept_time=0
    check_array=[0 for i in range(m)]
    correct = [0 if i <m/2 else 1 for i in range(m)]
    vote=[0 for _ in range(int(m))]
    for count,indice in zip(count_set,indices_set):
    
        keys=list(count.keys())
        keys=keys[0][::-1]

        print("ans: "+str(indice))
        print(keys)
        for i in range(len(keys)):
            if keys[i]=='0':
                vote[indice[i]]+=1
    print(f'vote: {vote}')
    for item in top_half_indices(vote):
        check_array[item]=1
    print(check_array)
    print()
    xor_result = [a ^ b for a, b in zip(check_array, correct)]
    accept_time=xor_result.count(0)
    return(accept_time) 


       
def check_outcome_condition_for_blended_case_special(counts_set,m,gate_num_times):
    accept_time=0
    vote=[0 for i in range(m)]
    check_array=[0 for i in range(m)]
    correct = [0 if i <m/2 else 1 for i in range(m)]
    n= int(np.log2(m))+1
    for key in counts_set:
        # print("origin: "+list(key)[0])
        raw_result=list(key)[0]
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(int(gate_num_times*m))]
        # print(result)
        result=[ int(item, 2) for item in result]
        # print(result)
        result=result[::-1] 
        
        print(result)

        for item in result:
            if item!=0:
                vote[item-1]+=1
        
    # print(vote)
    for item in top_half_indices(vote):
        check_array[item]=1
    print(check_array)
    print(correct)
    print()
    xor_result = [a ^ b for a, b in zip(check_array, correct)]
    accept_time=xor_result.count(0)
        
    return accept_time

def check_outcome_condition_for_interweave_case_special(counts_set,m,get_num_times):
    accept_time=0
    n= int(np.log2(m))+1
    check_array=[-1 for i in range(m)]
    correct = [0 if i <m/2 else 1 for i in range(m)]
    vote=[0 for _ in range(m) ]
    for key in counts_set:
        # print("origin: "+key)
        raw_result=list(key)[0]
        # print("after: "+raw_result)
        result=[raw_result[n*i:(i+1)*n] for i in range(int(get_num_times*m))]
        # print(result)
        result=[ int(item, 2) for item in result]
        # print(result)
        result=result[::-1] 
        
        print(result)
        
        
        for i in range(len(result)):
            if result[i]!=0:
                if i%2==0 and check_array[result[i]-1]==-1:
                    # check_array[result[i]-1]=0
                    vote[result[i]-1]-=1
                elif check_array[result[i]-1]==-1:
                    # check_array[result[i]-1]=1
                    vote[result[i]-1]+=1
    
    for i in range(len(vote)):
        if(vote[i]==0):
            check_array[i]=random.choice([0,1])
        elif(vote[i]>0):
            check_array[i]=1
        elif(vote[i]<0):
            check_array[i]=0
    print(vote)
    print(check_array)
    # print(check_array)
    xor_result = [a ^ b for a, b in zip(check_array, correct)]
    accept_time=xor_result.count(0)
    return accept_time


def check_outcome_condition_for_blended_three_case_special(counts_set,m,permutation,round,special=True):
    
    table=[0]*m
    for key in counts_set:
        keys=list(key)[0]
                        
        # print(keys)
        result=[ int(keys[2*i:(i+1)*2],2) for i in range(round)]
        result=result[::-1]
        
        print("result: "+ str(result))
        for i in range(len(result)):
            if result[i]==1:
                # print(f'permutation: {permutation[i]}')
                for j in range(len(permutation[i])):
                    
                    if permutation[i][j]==1:
                        table[j]+=1
            elif result[i]==2:
                # print(f'permutation: {permutation[i]}')
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
        print(table)
        
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        
        for item in top_half_indices(table):
            check_array[item]=1
        print(check_array)
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

def split_shadow_median(measurements, classical_shadow_set,K):
    
    # print(classical_shadow_set)
    N=len(classical_shadow_set)
    splited_shadow=np.array_split(classical_shadow_set, K)
    # print(splited_shadow)
    
    means_splited_shadow=[]
    
    for item in splited_shadow:
        sum=0
        for copy in item:
            sum+=copy
        # print(sum)
        mean=sum/(int(N/K))
        # print(mean)
        means_splited_shadow.append(mean)
    
    # for item in means_splited_shadow:
    
    #     print("trace :"+str(np.trace(abs(item))))
    probability_set_all_shadow=[[] for i in range(len(measurements)) ]
    
    for i in range(len(measurements)):
        for j in range(K):
            probability_set_all_shadow[i].append(abs(np.trace(measurements[i]@means_splited_shadow[j])))
    
    output=[]
    
    for item in probability_set_all_shadow:
        # print(item)
        median=np.median(item)
        # print(median)
        indices = np.where(item == median)[0]
        output.append(median)
    for item in output:
        print(item)
        
    return output