
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


def check_outcome_condition_for_random_case_special(test_time_count_set,test_time_indices_set,m,test_time):
    
   
    test_time_check_array=[]
    for i in range(test_time):
        check_array=[0 for i in range(m)]
        vote=[0 for _ in range(int(m))]
        for count,indice in zip(test_time_count_set[i],test_time_indices_set[i]):
        
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
        # print(check_array)
        test_time_check_array.append(check_array)
    print(test_time_check_array)
        
    correct = [0 if i <m/2 else 1 for i in range(m)]
    success_rate=0
    m_array=[[] for i in range(m)]
    for i, check_array in enumerate(test_time_check_array):
        for j in range(m):
            m_array[j].append(check_array[j])
    print(m_array)
    
    success_counter_set=[]
    for i,item in enumerate(m_array):
        success_counter=0 
        for j in item:
            if j==correct[i]:
                success_counter+=1
        success_counter_set.append(success_counter)
        print(f'success_counter: {success_counter}')
    success_counter = min(success_counter_set)
    success_rate=success_counter/test_time
    return(success_rate) 


       
def check_outcome_condition_for_blended_case_special(test_time_counts_set,m,gate_num_times,test_time):
    test_time_check_array=[]
    
    for i in range(test_time):
        vote=[0 for i in range(m)]
        check_array=[0 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        n= int(np.log2(m))+1
        for key in test_time_counts_set[i]:
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
        
        test_time_check_array.append(check_array)
    
    # print(test_time_check_array)
   
    correct = [0 if i <m/2 else 1 for i in range(m)]
    success_rate=0
    m_array=[[] for i in range(m)]
    for i, check_array in enumerate(test_time_check_array):
        for j in range(m):
            m_array[j].append(check_array[j])
    # print(m_array)
    
    success_counter_set=[]
    for i,item in enumerate(m_array):
        success_counter=0 
        for j in item:
            if j==correct[i]:
                success_counter+=1
        success_counter_set.append(success_counter)
        print(f'success_counter: {success_counter}')
    # success_counter = sum(success_counter_set) / len(success_counter_set)
    success_counter = min(success_counter_set)
    success_rate=success_counter/test_time
    return(success_rate) 
        

def check_outcome_condition_for_interweave_case_special(test_time_counts_set,m,get_num_times,test_time):
    
    n= int(np.log2(m))+1
    test_time_check_array=[]
    for i in range(test_time):
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        vote=[0 for _ in range(m) ]
        for key in test_time_counts_set[i]:
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
        
        test_time_check_array.append(check_array)
    # print(test_time_check_array)
    correct = [0 if i <m/2 else 1 for i in range(m)]
    success_rate=0
    m_array=[[] for i in range(m)]
    for i, check_array in enumerate(test_time_check_array):
        for j in range(m):
            m_array[j].append(check_array[j])
    # print(m_array)
    
    success_counter_set=[]
    for i,item in enumerate(m_array):
        success_counter=0 
        for j in item:
            if j==correct[i]:
                success_counter+=1
        success_counter_set.append(success_counter)
        print(f'success_counter: {success_counter}')
    # success_counter = sum(success_counter_set) / len(success_counter_set)
    success_counter = min(success_counter_set)
    success_rate=success_counter/test_time
    return(success_rate) 

def check_outcome_condition_for_blended_three_case_special(test_time_counts_set,m,permutation,round,test_time):
    
    test_time_check_array=[]
    for i in range(test_time):
        
        table=[0]*m
        for key in test_time_counts_set[i]:
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
    
        print(table)
        
        check_array=[-1 for i in range(m)]
        correct = [0 if i <m/2 else 1 for i in range(m)]
        
        for item in top_half_indices(table):
            check_array[item]=1
        
        for i in range(len(check_array)):
            if check_array[i]==-1:
                check_array[i]=0
        print(check_array)
        test_time_check_array.append(check_array)
    # print(test_time_check_array)

    correct = [0 if i <m/2 else 1 for i in range(m)]
    success_rate=0
    m_array=[[] for i in range(m)]
    for i, check_array in enumerate(test_time_check_array):
        for j in range(m):
            m_array[j].append(check_array[j])
    # print(m_array)
    
    success_counter_set=[]
    for i,item in enumerate(m_array):
        success_counter=0 
        for j in item:
            if j==correct[i]:
                success_counter+=1
        success_counter_set.append(success_counter)
        print(f'success_counter: {success_counter}')
    # success_counter = sum(success_counter_set) / len(success_counter_set)
    success_counter = min(success_counter_set)
    success_rate=success_counter/test_time
    return(success_rate) 
    
    

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
    # for item in output:
        # print(item)
        
    return output