from event_learning_fuc import event_learning
from tools import print_progress,generate_random_statevector
import numpy as np
import matplotlib.pyplot as plt
import os
############################### Initialization ######################################################

d = 2                             # Dimension of the initial state (need to be a power of 2)
m_s=[4,8,16,32,64]                  # the number of elements in the povm measurement
case_s=[1,2]                       # the case to test
state=np.array([[0,1]])           # initial state     
num_shot=5000                    # the shot for sampling in one circuit
test_time=1000                   # the number of times to run the circuit

event_learning_times=50          # run event learning several times 

projector=np.array([[1,0],[0,0]])

state_random=True
projector_random=False

dir_name="./random_plot/"
if not os.path.exists(dir_name):
        os.makedirs(dir_name)

############################# random state ###########################################################


if state_random:
    state=np.array([generate_random_statevector(d)])


############################# multi-run ###############################################################

for case in case_s:
    for m in m_s:
        y_thm=[]
        y_exp=[]
        for i in range(event_learning_times):
            result=event_learning(d,m,case,state,num_shot,test_time,projector,projector_random=projector_random)
            y_thm.append(result['theorem'])
            y_exp.append(result['experiemnt'])
            print_progress(i+1,event_learning_times,bar_length=event_learning_times)
        print()
        fig,ax=plt.subplots()
        # print(result)
        x=range(0,event_learning_times)
        if case==1:
            ax.plot(x,y_thm,label="theorem result (at least)")
        elif case==2:
            ax.plot(x,y_thm,label="theorem result (at most)")
        ax.plot(x,y_exp,label="experiment result")
        ax.set_title("Case"+str(case)+","+"m=" +str(m))
        ax.legend()
        plt.savefig(dir_name+"Case"+str(case)+","+"m=" +str(m)+".png")