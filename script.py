from event_learning_fuc import event_learning
from tools import print_progress
import numpy as np
import datetime
import json
# from plot import generate_plot    
if __name__=="__main__":

    
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%H_%M_%S")
    
    file_path="result_"+date_time+".json"
    
    with open('test_data.json', 'r') as file:
        data = json.load(file)
    
    case_1_high=data["case_1_high"]
    case_1_low=data["case_1_low"]
    case_2_pro=data["case_2_pro"]
    test_time=data["test_time"]
    average_time=data["average_time"]
    
    
    
    for test_data in data["test_data"]:
        state=np.array([np.hstack((1,np.zeros(test_data["d"]-1)))])           
        roh_0=np.outer(state,state.T.conj())

        povm_set=[]
        
        method=test_data["method"]
        case= test_data["case"]
        
        d=test_data["d"]
        m=test_data["m"]
        rank=test_data["rank"]
        
        gate_num_time=test_data["gate_num_time"]
        
        if method=="special_blended" or method=="special_random" or method=="interweave" or method=="blended_three":
            povm_set_m=np.load('./measurement_dir/special'+'_d_'+str(d)+'_m_'+str(m)+"_r_"+str(rank)+'.npy')
        else:
            povm_set_m=np.load('./measurement_dir/case_'+str(case)+'_d_'+str(d)+'_m_'+str(m)+"_r_"+str(rank)+'.npy')
            
        experiment_raw_data=[]
        theorem_raw_data=[]
        for i in range(average_time):
            result=event_learning(d,m,gate_num_time,povm_set_m[i],roh_0,case,state,test_time,rank,case_1_high,case_1_low,case_2_pro/m,method)
            print(f'result:{result}')
            experiment_raw_data.append(result["experiment"])
            theorem_raw_data.append(result["theorem"])
            print("\n"+str(result['experiment']))
            print_progress(i+1,average_time,bar_length=average_time)
            print()
        test_data["result"]={}
        test_data["result"]["experiment"]=np.mean(experiment_raw_data)
        test_data["result"]["theorem"]=np.mean(theorem_raw_data)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

   