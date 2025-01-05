import random
import numpy as np
import matplotlib.pyplot as plt


def generate_list(n):
    if n % 2 != 0:
        raise ValueError("n must be an even number")
    half = n // 2
    return [0] * half + [1] * half




def generate_success_pro(n,test_times,success_pro):

    result = generate_list(n)
    # print(result)  # Output: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    ans=[]

    for _ in range(test_times):
        ans.append([random.randint(0,1) for _ in range(n)])    
        
    # print(ans)
    n_array=[[] for i in range(n)]
    for i, check_array in enumerate(ans):
        for j in range(n):
            n_array[j].append(check_array[j])
    # print(n_array)

    success_counter_set=[]
    for i,item in enumerate(n_array):
        success_counter=0 
        for j in item:
            if j==result[i]:
                success_counter+=1
        success_counter_set.append(success_counter)
        # print(f'success_counter: {success_counter}')

    success_counter = min(success_counter_set)
    success_rate=success_counter/test_times
    success_pro.append(success_rate)
    
    return success_pro


m=[4,8,12,16,20,24]
success_pro=[]

for n in m:
    generate_success_pro(n,250,success_pro)


fig, ax = plt.subplots()
ax.plot(range(len(m)), success_pro, color='orange', label="experiment result")
ax.plot(range(len(m)), [0.5] * len(success_pro), label="50%")

ax.set_xlabel("number of measurement")
ax.set_ylabel("Success Probability")
ax.legend()
plt.xticks(range(len(m)), labels=m)
plt.yticks(np.arange(0, 1.2, 0.05))
plt.ylim(0, 1)
plt.title('Random Guessing')
plt.savefig('/home/jason/Documents/random_measurement/result_plot/success_probability_plot.png')
