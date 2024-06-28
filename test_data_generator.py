import json
from itertools import product


def generate_test_data(copies,d_s,m_s,rank_s,gate_num_times,method_s,case_1_high,case_1_low,case_2_pro,test_time,average_time,case):
    data={"test_data":[]}
    file_path="test_data.json"
    
    for d in d_s:
        for method,m,rank in product(method_s,m_s,rank_s[d]):
            
            if method =="interweave" or method=="special_blended":
                for gate_num_time in gate_num_times:
                    
                    data["test_data"].append({"d":d,
                                            "m":m,
                                            "rank":rank,
                                            "gate_num_time":gate_num_time,
                                            "method":method,
                                            "case":case,
                                            })
            else:
                data["test_data"].append({"d":d,
                                            "m":m,
                                            "rank":rank,
                                            "gate_num_time":1,
                                            "method":method,
                                            "case":case,
                                            })
    data["case_1_high"]=case_1_high
    data["case_1_low"]=case_1_low
    data["case_2_pro"]=case_2_pro
    data["test_time"]=test_time
    data["method_s"]=method_s
    data["m_s"]=m_s
    data["d_s"]=d_s
    data["rank_s"]=rank_s
    data["gate_num_times"]=gate_num_times
    data["average_time"]=average_time
    data["copies"]=copies
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__=="__main__":

    d_s = [4]          
    m_s=[10]         
    case=1                     
    rank_s={2:[1],
            4:[2,3],
            8:[4],
            16:[7,8,9],
            32:[16]}            # the rank of the povms

    test_time=50                     # the number of times to run the circuit
    average_time=30           # run event learning several times (each is indipendnet)
        
    copies=2

    case_1_high=0.9                 # the lower bound of the high accepting probability povm in case 1
    case_1_low=0.1                 # the upper bound of the low accepting probability povm in case 1 (/m)
    case_2_pro=0.01 #/m              # the upper bound of the low accepting probability povm in case 2 (/m)

    gate_num_times=[1]

    method_s=["special_blended","interweave","blended_three"]
    
    generate_test_data(copies,d_s,m_s,rank_s,gate_num_times,method_s,case_1_high,case_1_low,case_2_pro,test_time,average_time,case)