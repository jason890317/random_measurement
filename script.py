from event_learning_fuc import event_learning
from tools import print_progress,generate_random_statevector,generate_random_projector, generate_rank_n_projector
import numpy as np
import matplotlib.pyplot as plt
import os
############################### Initialization ######################################################

d = 32                             # Dimension of the initial state (need to be a power of 2)
m_s=[8,16,32]                  # the number of elements in the povm measurement
case_s=[1]                       # the case to test
          
# num_shot=1                  # the shot for sampling in one circuit
test_time=50                   # the number of times to run the circuit
rank_case_1_high= 28
rank_case_1_low= 8 
rank_case_2= 4                    # the rank of the projector
event_learning_times=10            # run event learning several times 

state_random=True                 # generate the random state
                                #generate the random projector to be the base of the povm

dir_name="d_32_case_1"       # set the directory saving the plot
if not os.path.exists(dir_name):
        os.makedirs(dir_name)

############################# random state ###########################################################


if state_random:
    state=np.array([generate_random_statevector(d)])
else:
    state=np.array([[0,1]])            # manually initialize the state

############################# multi-run ###############################################################

for case in case_s:
    for m in m_s:
        print("case: "+str(case)+", m: "+str(m))
        y_thm=[]
        y_exp=[]
        for i in range(event_learning_times):
            result=event_learning(d,m,case,state,test_time,rank_case_1_high,rank_case_1_low,rank_case_2)
            y_thm.append(result['theorem'])
            y_exp.append(result['experiemnt'])
            print_progress(i+1,event_learning_times,bar_length=event_learning_times)
            print()
        fig,ax=plt.subplots()
        # print(result)
        x=range(0,event_learning_times)
        if case==1:
            ax.plot(x,y_thm,label="theorem result (at least), rank_high= "+str(rank_case_1_high)+", rank_low= "+str(rank_case_1_low))
        elif case==2:
            ax.plot(x,y_thm,label="theorem result (at most), rank= "+str(rank_case_2))
        ax.plot(x,y_exp,label="experiment result")
        ax.set_title("Case"+str(case)+","+"m=" +str(m))
        ax.legend()
        plt.savefig("./"+dir_name+"/"+"Case"+str(case)+","+"m=" +str(m)+".png")