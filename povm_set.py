
from generate_povm import generate_povm_epson_case_1,generate_povm_epson_case_2,generate_povm_epson_case_special
import numpy as np
import json
# from plot import generate_plot    
############################### Initialization ######################################################



if __name__=="__main__":

    with open('test_data.json', 'r') as file:
        data = json.load(file)
   
    average_time=data["average_time"]
    case_1_high=data["case_1_high"]
    case_1_low=data["case_1_low"]
    case_2_pro=data["case_2_pro"]
    
    for test_data in data["test_data"]:
        state=np.array([np.hstack((1,np.zeros(test_data["d"]-1)))])           
        roh_0=np.outer(state,state.T.conj())

        povm_set=[]
        
        method=test_data["method"]
        case= test_data["case"]
        
        d=test_data["d"]
        m=test_data["m"]
        rank=test_data["rank"]
        
        
        if (method=="blended" or method=="random") and case==1:
                    
            for i in range(average_time):
                povm_set.append(generate_povm_epson_case_1(d,m,rank,case_1_high,case_1_low,roh_0))
            np.save('./measurement_dir/case_1_d_'+str(d)+'_m_'+str(m)+"_r_"+str(rank)+'.npy', povm_set)
            
        
        elif (method=="blended" or method=="random") and case==2:
            
            for i in range(average_time):
                povm_set.append(generate_povm_epson_case_2(d,m,rank,case_2_pro,roh_0))
            np.save('./measurement_dir/case_2_d_'+str(d)+'_m_'+str(m)+"_r_"+str(rank)+'.npy', povm_set)
            
        
        elif (method=='special_random' or method=="special_blended" or method=="interweave"):
            
            for i in range(average_time):
                povm_set.append(generate_povm_epson_case_special(d,m,rank,case_1_high,case_1_low,roh_0))
            np.save('./measurement_dir/special_d_'+str(d)+'_m_'+str(m)+"_r_"+str(rank)+'.npy', povm_set)

        
