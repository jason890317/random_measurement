from generate_povm import generate_povm_epson_case_1
from freedom_measurement import generate_freedom_measurement
import numpy as np
from scipy.linalg import sqrtm
from circuit import construct_blended_three_outcome_circuit,test_blended_circuit
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from tools import top_half_indices
# from plot import generate_plot

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
        

d_s = [16]          
m_s=[50,60,70,80]         
case=1                       
rank_s={2:[1],
        4:[2,3],
        8:[4],
        16:[8],
        32:[13,14,15,16,17,18,19]}         

method="blended_three"

pro_h=0.9
pro_l=0.1


plot=True
special=True

# number_of_high=10    
average_time=45
test_time=1  #if special => 1



dirname="./"+method
if not os.path.exists(dirname):
        os.makedirs(dirname)
        
d_y_value=[]
for d in d_s:
    
    state=np.array([np.hstack((1,np.zeros(d-1)))])           
    roh_0=np.outer(state,state.T.conj())    
    m_y_value=[]
    m_y_thm=[]
    for m in m_s:
        y_total=[]
        y_thm=[]
        
        for rank in rank_s[d]:
            
            round=m*5
        
            
            if special:
                povm_set_m=np.load('./measurement_dir/special_d_'+str(d)+'m_'+str(m)+"r_"+str(rank)+'.npy')
            else:
                povm_set_m=np.load('./measurement_dir/case_1_d_'+str(d)+'m_'+str(m)+"r_"+str(rank)+'.npy')
            
            average=[]
            
            ############## get one spot on the plot ####################
            
            
            # print("start to test")
            for average_itr in range(average_time):
                
                # print(average_itr)
                
                accept=0
                
                povm_set=povm_set_m[average_itr]
                
                # t=povm_set[number_of_high-1]
                # povm_set[number_of_high-1]=povm_set[m-1]
                # povm_set[m-1]=t
                
                ########### get one probability ####################
                
                special_average=[]
                
                for test_itr in range(test_time):
                    table=[0]*m
                    permutation=generate_permutations(m,round)
                    permutation = [list(t) for t in permutation]
                    # print(permutation)
                    # print("test: "+str(test_itr))
                    
                    three_outcome_blended_set=[]
                    
                    sum_set=np.zeros((d, d), dtype=np.complex128)
                    for item in povm_set:
                        sum_set+=item
                    identity=np.eye(d)
                    E_0=sqrtm(identity-sum_set/m)
                    E_0=E_0.astype('complex128')
                    
                    for round_itr in range(round):
                        sum_set_1=np.zeros((d, d), dtype=np.complex128)
                        sum_set_2=np.zeros((d, d), dtype=np.complex128)
                        # print(permutation[round_itr])
                        E=[]
                        E.append(E_0)
                        for i in range(m):
                            if permutation[round_itr][i]==1:
                                sum_set_1+= povm_set[i]
                            elif permutation[round_itr][i]==2:
                                sum_set_2+=povm_set[i]
                        
                        temp_1=sqrtm(sum_set_1/int(m))
                        temp_1=temp_1.astype('complex128')
                        temp_2=sqrtm(sum_set_2/int(m))
                        temp_2=temp_2.astype('complex128')
                        E.append(temp_1)
                        E.append(temp_2)
                        blended_set=[ item@item.T.conj() for item in E]
                        # print(E)
                        three_outcome_blended_set.append(blended_set)
                    
                    # for item in three_outcome_blended_set:
                    #     show_probability_povm(item,roh_0,True)
                    #     print()
                    qc=construct_blended_three_outcome_circuit(three_outcome_blended_set,state,round,m)
                    counts=test_blended_circuit(qc,1)
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
                        special_average.append(accept/m)
                if not special:
                    average.append(accept/test_time)
                else:
                    print(special_average)
                    average.append(np.mean(special_average))
                #####################################################
            y_thm.append(0)
            y_total.append(np.mean(average))
        m_y_thm.append(y_thm)
        m_y_value.append(y_total)
    np.savetxt("./"+dirname+"/"+"d="+str(d)+"_exp", m_y_value, fmt='%f')
    np.savetxt("./"+dirname+"/"+"d="+str(d)+"_thm", m_y_thm, fmt='%f')

# if plot:
#     generate_plot(d_s,m_s,case,rank_s,method)