from event_learning_fuc import event_learning
import numpy as np
from tools import print_progress
import matplotlib.pyplot as plt

############################### Initialization ######################################################

d = 2                             # Dimension of the initial state
m_s=[4,8,16,32,64,128]                  # the number of elements in the povm measurement
case_s=[1,2]                       # the case to test
state=np.array([[1,0]])           # initial state     
num_shot=5000                    # the shot for sampling in one circuit
test_time=1000                   # the number of times to run the circuit

event_learning_times=50          # run event learning several times 



############################# multi-run ###############################################################

for case in case_s:
    for m in m_s:
        y_thm=[]
        y_exp=[]
        for i in range(event_learning_times):
            result=event_learning(d,m,case,state,num_shot,test_time,rotation=True,plot=False,print_check=False)
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
        plt.savefig("./plot/"+"Case"+str(case)+","+"m=" +str(m)+".png")