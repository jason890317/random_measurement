import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from tools import print_eigenvalue, show_probability_povm,generate_binary_strings,resolve_blended_result_case_1,resolve_blended_result_case_2
from generate_povm import generate_povm_set_case_1,generate_povm_set_case_2, generate_povm_by_unitary_case_1, generate_povm_by_unitary_case_2,generate_povm_epson_case_1,generate_povm_epson_case_2
from blended_measurement import blended_measurement
from circuit import construct_circuit_and_test,construct_blended_circuit,test_blended_circuit


def event_learning(d,m,case,state,test_time,rank,epson_case_1,epson_case_2,epson_rotation=True,plot=False,print_check=False):
    

    ################### Initialization ######################################################

    roh_0=np.outer(state,state.T.conj())                # cauculate the density matrix
    state = state/np.linalg.norm(state)                 # normalize the state
    ################### generate the povm measurement sets ##################################

    if case==1:
        if epson_rotation:
            povm_set=generate_povm_epson_case_1(d,m,rank,epson_case_1,roh_0)
        # else:
        #     povm_set = generate_povm_by_unitary_case_1(d,m,rank_case_1_high,rank_case_1_low,roh_0)
        
    elif case==2:
        if epson_rotation:
            povm_set=generate_povm_epson_case_2(d,m,rank,epson_case_2,roh_0)
        # else:
        #     povm_set =generate_povm_by_unitary_case_2(d,m,rank_case_2,roh_0)
    ################### cauculate the probability of elements in povm set #################################################

    # print_eigenvalue(povm_set)
    povm_set_pro=show_probability_povm(povm_set,roh_0,False)

    ################# test the accepting probability of povm Mi (simulation)#################################################

    
    # result_povm=[]

    # for item in povm_set:
    #     povm=[]
    #     povm.append(item)
    #     inverse=np.eye(d)-item
    #     povm.append(inverse)    
        
    #     counts=construct_circuit_and_test(povm,state,num_shot)
    #     if '0' not in counts.keys():
    #         result_povm.append(0)
    #     else:
    #         result_povm.append(counts['0'])

    # result_povm=[item/num_shot for item in result_povm]
    # if plot:
    #     fig, ax = plt.subplots()
    #     ax.plot(range(0,m),result_povm)
    #     ax.set_xlabel(" two outcome povm ")
    #     ax.set_ylabel(" accepting probability ")
    #     ax.set_title(" the accepting probability for each two outcome povm ")
    #     plt.show()


    ##################### cauculate the coefficients #################################

    povm_set_pro=sorted(povm_set_pro,reverse=True)

    if case==1:
        
        # epsilon=float(1-povm_set_pro[0])
        # beta=float(sum(povm_set_pro[1:]))
        # at_least_pro=float(((1-epsilon)**3)/(12*(1+beta)))
        epsilon=1-povm_set_pro[0]
        beta=sum(povm_set_pro[1:])
        at_least_pro=((1-epsilon)**3)/(12*(1+beta))
        if print_check:
            print("   Case 1")
            print("1. epsilon is "+str(epsilon))
            print("2. beta is "+str(beta))
            print("3. At least accepting probability is "+str(at_least_pro))

    elif case==2:
        
        # delta=float(sum(povm_set_pro[:]))
        delta=sum(povm_set_pro[:])
        if print_check:
            print("   Case 2")
            print("1. delta is "+str(delta))
            
    ################## generate blended measurement sets #############################################

    blended_set=blended_measurement(povm_set,d,m)
    blended_set=[ item@item.T.conj() for item in blended_set]

    ################# test the accepting probability of blended measurement Ei (simulation) #################################################

    # counts=construct_circuit_and_test(blended_set,state,test_time)

    # result_blended=[]

    # set_num=generate_binary_strings(int(np.log2(m))+1)
    # set_num=np.array(list(sorted(set_num)))

    # cut=int((len(set_num)/2)+1)

    # for item in set_num[0:cut]:
    #     if item in counts.keys():
    #         result_blended.append(counts[item]/test_time)
    #     else:
    #         result_blended.append(0)
    # print(result_blended)
    # if plot:
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel("blended measurement outcome")
    #     ax.set_ylabel("accepting probability")
    #     ax.set_title("blended measurement result")
    #     ax.plot(range(0,m+1),result_blended)
    #     plt.show()

    ############### Sequential Blend Measurement impletement(simulation) #########################
    counts_set=[]
    qc=construct_blended_circuit(blended_set,state,m)
    
    ####### without parallelization ##########
    # for i in range(test_time):
    #     print(f'\r{i}', end='', flush=True)
    #     counts=test_blended_circuit(qc,1)
    #     counts_set.append(counts)
   
    ###### with parallelization but not batch
    # counts_set=test_blended_circuit(qc,test_time)
    
    ###### batch ############################
    for i in range(int(test_time/10)):
        print(f'\r{i}', end='', flush=True)
        counts=test_blended_circuit(qc,10)
        for item in counts.items():
            counts_set.append(item)
    counts_set=dict(counts_set)

   
    # print(counts_set)
    #################### Dealing with one sequential blended measurement result #########################

    # count the appearing time of each outcome
    # number_counts=resolve_blended_result(counts,m)
    # labels, values = zip(*number_counts.items())

    # Create the plot
    # plot_sequential_blended_result(labels,values,m)


    ################## Check the theorem and the experiment result ######################################
    accept_time=0
    if case == 2:
        # for count in counts_set:
        #     number_counts=resolve_blended_result_case_2(count,m)
        #     # print(number_counts)
        #     labels, values = zip(*number_counts.items())
        #     if len(labels)>1:
        #         accept_time+=1
        # experiment=accept_time/test_time
        # if print_check:
        #     print("The probability of getting  accept at least one time: "+str(experiment)+"\n"+"The probability at most: "+ str(delta))
        accept_time=resolve_blended_result_case_2(counts_set,m)
        experiment=accept_time/test_time
    elif case == 1:
        # for count in counts_set:
        #     print(count)
        #     if resolve_blended_result_case_1(count,m):
        #         accept_time+=1
        # print(accept_time)
        # experiment=accept_time/test_time
        # if print_check:
        #     print("The probability of getting the accept at least one time and the accept with high accepting probability: "+str(experiment)+"\n"+"The probability at least: "+str(at_least_pro))
        accept_time=resolve_blended_result_case_1(counts_set,m)
        experiment=accept_time/test_time
    ################ Return the result ################################################################
    
    if case==1:
        result={"theorem":at_least_pro,"experiemnt":experiment}
    elif case==2:
        result={"theorem":delta,"experiemnt":experiment}
    
    return result