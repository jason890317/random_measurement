import numpy as np
from tools import (
    show_probability_povm, resolve_blended_result_case_1, resolve_blended_result_case_2,
    resolve_random_result_case_special, resolve_random_result_case_1, resolve_blended_result_case_special,
    resolve_blended_result_case_interweave, generate_permutations, resolve_blended_three_result,
    resolve_random_result_case_2
)
from blended_measurement import (
    blended_measurement, inverse_blended_measurement, optimizing_blended_measurement, three_outcome_blended_measurement
)
from circuit import (
    blended_circuit, random_sequences_circuit, run_circuit,
    interweave_blended_circuit, three_outcome_blended_circuit
)

def event_learning(copies,d,m,gate_num_times,povm_set,roh_0,case,state,test_time,case_1_high,method):
    
  
    if method=='special_random':
        # povm_set=generate_povm_epson_case_special(d,m,rank,pro_case_1_h,pro_case_1_l,roh_0)
        count_set=[]
        accept_time=0
        for i in range(copies):
            qc,high=random_sequences_circuit(povm_set,state,m,case_1_high)
            # print(high)
            counts=run_circuit(qc,num_shot=1,backend='qasm_simulator')
            
            count_set.append([counts,high])
        # print(counts)
        accept_time=resolve_random_result_case_special(count_set,m)
        
        experiment=accept_time/m
    if method=='special_blended' or method=="optimizing_blended":
        # povm_set=generate_povm_epson_case_special(d,m,rank,pro_case_1_h,pro_case_1_l,roh_0)
        
        # povm_set_pro=show_probability_povm(povm_set,roh_0,True)
        accept_time=0
        
        if method=="optimizing_blended":
            blended_set=optimizing_blended_measurement(povm_set,d,m)
        elif method=="special_blended":
            blended_set=blended_measurement(povm_set,d,m)
        
        # blended_set=[ item@item.T.conj() for item in blended_set]
        # povm_set_pro=show_probability_povm(blended_set,roh_0,False)
        
        counts_set=[]
        
        for _ in range(copies):    
            qc=blended_circuit(blended_set,state,int(gate_num_times*m))
            
            # print(high)
            counts=run_circuit(qc,num_shot=1,backend='qasm_simulator')
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
        # blended_set=[ item@item.T.conj() for item in blended_set]
        blended_set_inv=inverse_blended_measurement(povm_set,d,m)
        # blended_set_inv=[ item@item.T.conj() for item in blended_set_inv]
        
        # povm_set_pro=show_probability_povm(blended_set,roh_0,False)
        counts_set=[]
        for _ in range(copies):
            qc=interweave_blended_circuit(blended_set,blended_set_inv,state,int(gate_num_times*m))
            # print(high)
            counts=run_circuit(qc,num_shot=1,backend='qasm_simulator')
            #print(counts)
            counts_set.append(counts)
            
        accept_time=resolve_blended_result_case_interweave(counts_set,m,gate_num_times)
         
        experiment=accept_time/m   
    
    #######################################################################################################################
    
    if method == "blended_three":
        
        
        permutation=generate_permutations(m,int(5*m))
        three_outcome_blended_set=three_outcome_blended_measurement(povm_set,d,m,permutation)
        # for item in three_outcome_blended_set:
        #     show_probability_povm(item,roh_0,True)
        #     print()
        counts_set=[]
        for _ in range(copies):
            qc=three_outcome_blended_circuit(three_outcome_blended_set,state,int(5*m))
            counts=run_circuit(qc,1)
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
        if case==1:
            for i in range(test_time):
                
                qc,high=random_sequences_circuit(povm_set,state,m,case_1_high)
                counts=run_circuit(qc,num_shot=1,backend='qasm_simulator')
                check=resolve_random_result_case_1(counts,high)
                if check:
                    accept_time+=1
            experiment=accept_time/test_time
        
        elif case==2:
            for i in range(test_time):
                qc,_=random_sequences_circuit(povm_set,state,m,case_1_high)
                counts=run_circuit(qc,num_shot=1,backend='qasm_simulator')
                check=resolve_random_result_case_2(counts)
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
   
        
        #############################################################################
        
        counts_set=[]
        
        qc=blended_circuit(blended_set,state,m)
        # print(povm_set)
        for i in range(int(test_time/50)):
            # print(f'\r{i}', end='', flush=True)
            counts=run_circuit(qc,50)
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
    elif method=='special_random' or method=='special_blended' or method=="interweave" or method=="blended_three" or method=="optimizing_blended":
        result={"theorem":0,"experiment":experiment}
        
    return result