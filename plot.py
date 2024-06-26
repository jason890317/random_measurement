import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
import pandas as pd
import sys

if __name__=="__main__":
    
    
    test_data_json=sys.argv[1]
    save_dir="./result_plot"
    
    with open(test_data_json, 'r') as file:
        data = json.load(file)
        
    
        
    
    pd_data=pd.DataFrame(data["test_data"])
    
    
    dir_name_s=save_dir+"/"+pd_data["method"].unique()
    
    for dir_name in dir_name_s:
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    
    
    
    ################################## rank plot ###########################################################
    
    grouped = pd_data.groupby(['d', 'm', 'method', 'gate_num_time'])
    for name,group in grouped:
        # print("Group:", name)
        
        exp=[]
        thm=[]
        rank_s=[]
        for item in group.values:
            exp.append(item[6]["experiment"])
            thm.append(item[6]["theorem"])
            rank_s.append(item[2])

        fig,ax=plt.subplots()
        plt.xticks(range(len(rank_s)),labels=rank_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        ax.plot(range(len(rank_s)),exp,color='orange',label="experiment result")
        ax.plot(range(len(rank_s)),thm,color='blue',label="theorem result")
        ax.set_title("d_"+str(name[0])+",cof_"+str(name[3])+",m_"+str(name[1]))
        ax.set_xlabel("the ranks")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+name[2]+"/d_"+str(name[0])+"gate_number_cof"+str(name[3])+",m_"+str(name[1])+".png")
        
    ################################ m plot ##############################################################
    
    grouped = pd_data.groupby(['d', 'method', 'gate_num_time'])
    for name,group in grouped:
        
        exp=[]
        thm=[]
        m_s=[]
        
        
        m_group=group.groupby('m')
        
        for sub_name, sub_group in m_group:
            
            m_s.append(sub_name)
            take_average_exp=[]
            take_average_thm=[]
            for item in sub_group.values:
                
                take_average_exp.append(item[6]['experiment'])
                take_average_thm.append(item[6]['theorem'])
            
            exp.append(np.mean(take_average_exp))
            thm.append(np.mean(take_average_thm))
        
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        
        if name[1]=="special_blended" or name[1]=="special_random" or name[1]=="interweave" or name[1]=="blended_three":
            ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%")
            plt.ylim(0, 1)
        if name[1]=="blended" or name[1]=='random':
            ax.plot(range(len(m_s)),thm,color='blue',label="theorem result")
        ax.plot(range(len(m_s)),exp,color='orange',label="experiment result")
        
        ax.set_title("d_"+str(name[0])+",cof_"+str(name[2]))
        ax.set_xlabel("the number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+name[1]+"/d_"+str(name[0])+"gate_number_cof"+str(name[2])+".png")
     
    ####################################### method plot ############################################################   
    filtered_df = pd_data[(pd_data["gate_num_time"]==1.0) &((pd_data['method'] == 'blended_three') | (pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') | (pd_data['method'] == 'special_random'))]
    
    grouped = filtered_df.groupby([ 'd'])
    
    
    for name,group in grouped:
        # print(group)
        method_plot=[]
        method=[]
        d_group=group.groupby(['method', 'gate_num_time'])
        for sub_name,sub_group in d_group:
            # print(sub_group)
            # print()
            exp=[]
            m_s=[]
            m_group=sub_group.groupby('m')
            
            for sub_sub_name, sub_sub_group in m_group:
                  
                m_s.append(sub_sub_name)
                take_average_exp=[]
                take_average_thm=[]
                for item in sub_sub_group.values:
                    
                    take_average_exp.append(item[6]['experiment'])
                    take_average_thm.append(item[6]['theorem'])
                
                exp.append(np.mean(take_average_exp))
            
            method_plot.append(exp)
            method.append(sub_name[0])
    
         
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        
        
        ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%")
        plt.ylim(0, 1)
        
        for method_name, plot in zip(method,method_plot):
            ax.plot(range(len(m_s)),plot,label=method_name)
        
        ax.set_title("d_"+str(name[0]))
        ax.set_xlabel("the number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"d_"+str(name[0])+"_cof_1.0"+".png")
    
    ########################################## gate num time plot ####################################################
    filtered_df = pd_data[(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')]
    
    grouped = filtered_df.groupby([ 'd'])
    
    
    for name,group in grouped:
        # print(group)
        
        d_group=group.groupby(['method'])
        for sub_name,sub_group in d_group:
           
            
            # print()
            gate_num_plot=[]
            gate_num=[]
            gate_num_group=sub_group.groupby(['gate_num_time'])
            for sub_sub_name,sub_sub_group in gate_num_group:
                # print(sub_sub_group)
                # print()
                exp=[]
                m_s=[]
                m_group=sub_sub_group.groupby('m')
                
                for sub_sub_sub_name, sub_sub_sub_group in m_group:
                    
                    m_s.append(sub_sub_sub_name)
                    take_average_exp=[]
                    take_average_thm=[]
                    for item in sub_sub_sub_group.values:
                        
                        take_average_exp.append(item[6]['experiment'])
                        take_average_thm.append(item[6]['theorem'])
                    
                    exp.append(np.mean(take_average_exp))
                
                gate_num_plot.append(exp)
                gate_num.append(sub_sub_name[0])

            
         
            fig,ax=plt.subplots()
            plt.xticks(range(len(m_s)),labels=m_s)
            plt.yticks(np.arange(0, 1.2, 0.05))
            
            
            ax.plot(range(len(m_s)),[0.5]*len(m_s),label="50%")
            plt.ylim(0, 1)
            
            for gate_name, plot in zip(gate_num,gate_num_plot):
                ax.plot(range(len(m_s)),plot,label="cof="+str(gate_name))
            
            ax.set_title("d_"+str(name[0]))
            ax.set_xlabel("the number of measurements")
            ax.set_ylabel("Success Probability")
            ax.legend()
            
            plt.savefig(save_dir+"/"+sub_name[0]+"/d_"+str(name[0])+".png")
    
    ####################################################################################################################
    
    cof_s=[0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8,2]
    for cof in cof_s:
        filtered_df = pd_data[(pd_data["gate_num_time"]==cof) &((pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') )]
    
        grouped = filtered_df.groupby([ 'd'])
        
        
        for name,group in grouped:
            # print(group)
            method_plot=[]
            method=[]
            d_group=group.groupby(['method', 'gate_num_time'])
            for sub_name,sub_group in d_group:
                # print(sub_group)
                # print()
                exp=[]
                m_s=[]
                m_group=sub_group.groupby('m')
                
                for sub_sub_name, sub_sub_group in m_group:
                    
                    m_s.append(sub_sub_name)
                    take_average_exp=[]
                    take_average_thm=[]
                    for item in sub_sub_group.values:
                        
                        take_average_exp.append(item[6]['experiment'])
                        take_average_thm.append(item[6]['theorem'])
                    
                    exp.append(np.mean(take_average_exp))
                
                method_plot.append(exp)
                method.append(sub_name[0])
        
            
            fig,ax=plt.subplots()
            plt.xticks(range(len(m_s)),labels=m_s)
            plt.yticks(np.arange(0, 1.2, 0.05))
            
            
            ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%")
            plt.ylim(0, 1)
            
            for method_name, plot in zip(method,method_plot):
                ax.plot(range(len(m_s)),plot,label=method_name)
            
            ax.set_title("d_"+str(name[0]))
            ax.set_xlabel("the number of measurements")
            ax.set_ylabel("Success Probability")
            ax.legend()
            
            plt.savefig(save_dir+"/"+"d_"+str(name[0])+"_cof_"+str(cof)+".png")
    
    ##################################################################################################################
    
    m=90
    
    filtered_df = pd_data[(pd_data["m"]==m) &((pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') )]

    # print(filtered_df)
    grouped = filtered_df.groupby(['d'])
    
    
    for name,group in grouped:
        # print(group)
        method_plot=[]
        method=[]
        d_group=group.groupby(['method'])
        for sub_name,sub_group in d_group:
            # print(sub_group)
            # print()
            exp=[]
            cof_s=[]
            cof_group=sub_group.groupby('gate_num_time')
            
            for sub_sub_name, sub_sub_group in cof_group:
                
                # print(sub_sub_name)
                cof_s.append(sub_sub_name)
                take_average_exp=[]
                take_average_thm=[]
                for item in sub_sub_group.values:
                    
                    take_average_exp.append(item[6]['experiment'])
                    take_average_thm.append(item[6]['theorem'])
                
                exp.append(np.mean(take_average_exp))
            
            method_plot.append(exp)
        
            method.append(sub_name[0])

        # print(method_plot)
        
        fig,ax=plt.subplots()
        plt.xticks(range(len(cof_s)),labels=cof_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        
        
        ax.plot(range(len(cof_s)),[0.5]*len(exp),label="50%")
        plt.ylim(0, 1)
        
        for method_name, plot in zip(method,method_plot):
            ax.plot(range(len(cof_s)),plot,label=method_name)
        
        ax.set_title("m_"+str(m))
        ax.set_xlabel("the different cofs")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"cofs_m_"+str(m)+".png")