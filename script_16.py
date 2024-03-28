from event_learning_fuc import event_learning
from tools import print_progress,generate_random_statevector,generate_random_projector, generate_rank_n_projector
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
############################### Initialization ######################################################

d = 16                           # Dimension of the initial state (need to be a power of 2)
m_s=[10,30,50]                  # the number of elements in the povm measurement
case_s=[1,2]                       # the case to test
rank_s=[4,8,12]
# num_shot=1                  # the shot for sampling in one circuit
test_time=500              # the number of times to run the circuit
event_learning_times=10            # run event learning several times 
standard_deviation_num=5

# rank_case_1_high= 4
# rank_case_1_low= 2 
# rank_case_2= 2                    # the rank of the projector


pro_case_1_h=0.9
pro_case_1_l=0.1 #/m
pro_case_2=0.01 #/m 
state_random=False               # generate the random state

top_num=6

                                #generate the random projector to be the base of the povm
for m in m_s:
    dir_name="d_"+str(d)+"_m_"+str(m)       # set the directory saving the plot
    if not os.path.exists(dir_name):
            os.makedirs(dir_name)
for case in case_s:
    for m in m_s:
        for rank in rank_s:
            with open("d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(case)+"_projector.html", 'w+') as file:
                    file.write("<html><body>")
############################# random state ###########################################################


if state_random:
    state=np.array([generate_random_statevector(d)])
else:
    state=np.array([np.hstack((1,np.zeros(d-1)))])           # manually initialize the state
############################# multi-run ###############################################################

for case in case_s:
    for rank in rank_s:
        for m in m_s:
            y_thm=[]
            y_exp=[]
            
            print("case: "+str(case)+", m: "+str(m))
            for _ in range(event_learning_times):
                y_temp=[]
                for i in range(standard_deviation_num):
                    result=event_learning(d,m,case,state,test_time,rank,pro_case_1_h,pro_case_1_l/m,pro_case_2/m,top_num,epson_rotation=True)
                    y_temp.append(result['experiment'])
                    print("\n"+str(result['experiment']))
                    print_progress(i+1,standard_deviation_num,bar_length=standard_deviation_num)
                    print()
                # print(y_temp)
                y_thm.append(result['theorem'])
                y_exp.append(y_temp)
                print_progress(_+1,event_learning_times,bar_length=event_learning_times)
                print()
                
            with open("d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(case)+"_projector.html", 'a') as file:
                file.write("</body></html>")


            means = np.mean(y_exp, axis=1)
            sem = stats.sem(y_exp, axis=1)
            confidence = 0.95
            ci = sem * stats.t.ppf((1 + confidence) / 2., standard_deviation_num - 1)
            lower_bound=[]
            for i in range(event_learning_times):
                if means[i]-ci[i]<0:
                    lower_bound.append(0)
                else:
                    lower_bound.append(means[i]-ci[i])
            fig,ax=plt.subplots()
            # print(result)
            x=range(0,event_learning_times)
            if case==1:
                ax.plot(x,y_thm,label="theorem result (at least)")
                plt.xticks(np.arange(1,event_learning_times , 1))
                plt.yticks(np.arange(0, 0.3, 0.01))
                plt.fill_between(x, lower_bound, means + ci, color='blue', alpha=0.2, label='95% Confidence Interval')
            elif case==2:
                ax.plot(x,y_thm,label="theorem result (at most)")
                plt.xticks(np.arange(1,event_learning_times , 1))
                plt.yticks(np.arange(0,0.1 , 0.001))
                plt.fill_between(x, lower_bound, means + ci, color='blue', alpha=0.2, label='95% Confidence Interval')
            ax.plot(x,means,label="experiment result")
            ax.set_title("Dimension: "+str(d)+", Case"+str(case)+","+" m=" +str(m)+", rank= "+str(rank))
            ax.legend()
            plt.savefig("./"+"d_"+str(d)+"_m_"+str(m)+"/"+"d="+str(d)+"_Case="+str(case)+"_m=" +str(m)+"_r="+str(rank)+".png")