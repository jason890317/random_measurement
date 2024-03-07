#!/usr/bin/env python
# coding: utf-8

# In[7]:


from event_learning_fuc import event_learning
import numpy as np
from tools import print_progress
import matplotlib.pyplot as plt


# In[8]:


############################### Initialization ######################################################

d = 2                             # Dimension of the initial state
m = 4                            # Number of POVM elements
case = 2                         # the case to test
state=np.array([[1,0]])           # initial state     
num_shot=1000                    # the shot for sampling in one circuit
test_time=100                   # the number of times to run the circuit

event_learning_times=10           # run event learning several times 



# In[9]:


y_thm=[]
y_exp=[]

for i in range(event_learning_times):
    result=event_learning(d,m,case,state,num_shot,test_time,rotation=True,plot=False,print_check=False)
    y_thm.append(result['theorem'])
    y_exp.append(result['experiemnt'])
    print_progress(i+1,event_learning_times,bar_length=event_learning_times)
print()
fig,ax=plt.subplots()
x=range(0,event_learning_times)
ax.plot(x,y_thm,label="theorem result")
ax.plot(x,y_exp,label="experiment result")
ax.legend()
plt.plot()

