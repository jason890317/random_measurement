import numpy as np


from tools import show_probability_povm,resolve_blended_result_case_1,resolve_blended_result_case_2,resolve_random_result_case_special,resolve_random_result_case_1,resolve_blended_result_case_special, resolve_blended_result_case_interweave,generate_permutations,resolve_blended_three_result
from generate_povm import generate_povm_epson_case_1,generate_povm_epson_case_2,generate_povm_epson_case_special
from blended_measurement import blended_measurement,blended_measurement_inverse
from circuit import construct_blended_circuit,test_blended_circuit,random_sequences_circuit,test_random_circuit,construct_blended_circuit_inverse,construct_blended_three_outcome_circuit

from scipy.linalg import sqrtm

def event_learning(copies,d,m,gate_num_times,povm_set,roh_0,case,state,test_time,case_1_high,method):
    
  
    if method=='special_random':
        # povm_set=generate_povm_epson_case_special(d,m,rank,pro_case_1_h,pro_case_1_l,roh_0)
        
        accept_time=0
        
        qc,high=random_sequences_circuit(povm_set,state,m,case_1_high)
        # print(high)
        counts=test_random_circuit(qc,num_shot=1,backend='qasm_simulator')
        # print(counts)
        accept_time=resolve_random_result_case_special(counts,high)
        
        experiment=accept_time/m
    if method=='special_blended':
        # povm_set=generate_povm_epson_case_special(d,m,rank,pro_case_1_h,pro_case_1_l,roh_0)
        
        # povm_set_pro=show_probability_povm(povm_set,roh_0,True)
        accept_time=0
        
        blended_set=blended_measurement(povm_set,d,m)
        blended_set=[ item@item.T.conj() for item in blended_set]
        # povm_set_pro=show_probability_povm(blended_set,roh_0,False)
        
        counts_set=[]
        
        for _ in range(copies):    
            qc=construct_blended_circuit(blended_set,state,int(gate_num_times*m))
            
            # print(high)
            counts=test_random_circuit(qc,num_shot=1,backend='qasm_simulator')
            counts_set.append(counts)
            
        accept_time=resolve_blended_result_case_special(counts_set,m,gate_num_times)
        
        # accept_time=resolve_random_result_case_special(counts,high)
        
        experiment=accept_time/m
    
    if method=="interweave":
        # povm_set=generate_povm_epson_case_special(d,m,rank,pro_case_1_h,pro_case_1_l,roh_0)
        
        # povm_set_pro=show_probability_povm(povm_set,roh_0,True)
        accept_time=0
        # povm_set_pro=show_probability_povm(povm_set,roh_0,True)
        blended_set=blended_measurement(povm_set,d,m)
        blended_set=[ item@item.T.conj() for item in blended_set]
        blended_set_inv=blended_measurement_inverse(povm_set,d,m)
        blended_set_inv=[ item@item.T.conj() for item in blended_set_inv]
        
        # povm_set_pro=show_probability_povm(blended_set,roh_0,False)
        counts_set=[]
        for _ in range(copies):
            qc=construct_blended_circuit_inverse(blended_set,blended_set_inv,state,int(gate_num_times*m))
            # print(high)
            counts=test_random_circuit(qc,num_shot=1,backend='qasm_simulator')
            #print(counts)
            counts_set.append(counts)
            
        accept_time=resolve_blended_result_case_interweave(counts_set,m,gate_num_times)
         
        experiment=accept_time/m   
    
    #######################################################################################################################
    
    if method == "blended_three":
        
        
        permutation=generate_permutations(m,int(5*m))
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
        
        for round_itr in range(int(5*m)):
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
        counts_set=[]
        for _ in range(copies):
            qc=construct_blended_three_outcome_circuit(three_outcome_blended_set,state,int(5*m),m)
            counts=test_blended_circuit(qc,1)
            counts_set.append(counts)
            # print(counts)
        accept_time=resolve_blended_three_result(counts_set,m,permutation,int(5*m))
        experiment=accept_time/m
        
    
    
    ################### cauculate the probability of elements in povm set #################################################
   
    povm_set_pro=show_probability_povm(povm_set,roh_0,False)
    povm_set_pro=sorted(povm_set_pro,reverse=True)
    
    ##################### cauculate the coefficients #################################
    
    if method=='random':
        
        #############################################################################
        if case==1:
            
            epsilon=1-povm_set_pro[0]
            
            beta=sum(povm_set_pro[1:])
            
            at_least_pro=((1-epsilon)**7)/(1296*((1+beta))**7)
            
        elif case==2:
            
            delta=2*sum(povm_set_pro[:])
        
        
        ##############################################################################
        
        accept_time=0
        # print(povm_set)
        for i in range(test_time):
            
            qc,high=random_sequences_circuit(povm_set,state,m,case_1_high)
            counts=test_random_circuit(qc,num_shot=1,backend='qasm_simulator')
            check=resolve_random_result_case_1(counts,high)
            if check:
                accept_time+=1
        experiment=accept_time/test_time
        
        
    if method=='blended':
        
        ##############################################################################
        if case==1:
            
            epsilon=1-povm_set_pro[0]
            
            beta=sum(povm_set_pro[1:])
            
            at_least_pro=((1-epsilon)**3)/(12*(1+beta))
            
        elif case==2:
            
            delta=sum(povm_set_pro[:])
        
        #############################################################################
        
        
        blended_set=blended_measurement(povm_set,d,m)
        blended_set=[ item@item.T.conj() for item in blended_set]
        
        #############################################################################
        
        counts_set=[]
        
        qc=construct_blended_circuit(blended_set,state,m)
        # print(povm_set)
        for i in range(int(test_time/50)):
            # print(f'\r{i}', end='', flush=True)
            counts=test_blended_circuit(qc,50)
            for item in counts.items():
                counts_set.append(item)
        count_set_dict={}
        for item in counts_set:
            key = item[0]  # The outcome or measurement key
            count = item[1]  # The count of this particular outcome

            # Accumulate the counts in the dictionary
            if key in count_set_dict:
                count_set_dict[key] += count
            else:
                count_set_dict[key] = count

        ##############################################################################
        
        accept_time=0
        
        if case == 2:
        
            accept_time=resolve_blended_result_case_2(count_set_dict,m)
            experiment=accept_time/test_time
            
        elif case == 1:
            
            accept_time=resolve_blended_result_case_1(count_set_dict,m)
            experiment=accept_time/test_time

    ################ Return the result ################################################################
    
    if case==1 and (method=="blended" or method=="random"):
        result={"theorem":at_least_pro.real,"experiment":experiment}
    elif case==2 and (method=="blended" or method=="random"):
        result={"theorem":delta.real,"experiment":experiment}
    elif method=='special_random' or method=='special_blended' or method=="interweave" or method=="blended_three":
        result={"theorem":0,"experiment":experiment}
        
    return result