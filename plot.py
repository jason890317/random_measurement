import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import sys

###########################################################################3
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial



############################################################################




if __name__=="__main__":
    
    
    test_data_json=sys.argv[1]
    save_dir="./result_plot/"+test_data_json[12:-5]+"/"
    if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    
    
    with open(test_data_json, 'r') as file:
        data = json.load(file)
        
    pd_data=pd.DataFrame(data["test_data"])
    
    
    dir_name_s=save_dir+"/"+pd_data["method"].unique()
    
    for dir_name in dir_name_s:
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    
    
    
    ################################## rank plot ###########################################################
    # filtered_df = pd_data[(pd_data["gate_num_time"]==1) & (pd_data["copies"]==20) ]
    # grouped = filtered_df.groupby(['d', 'm', 'method', 'gate_num_time','copies'])
    
    
    # for name,group in grouped:
    #     # print("Group:", group)
    #     # print()
    #     exp=[]
    #     # thm=[]
    #     rank_s=[]
    #     for item in group.values:
    #         # print(item)
    #         exp.append(np.mean(item[7]["experiment"]))
    #         # thm.append(item[7]["theorem"])
    #         rank_s.append(item[2])
      
    #     fig,ax=plt.subplots()
    #     plt.xticks(range(len(rank_s)),labels=rank_s)
    #     plt.yticks(np.arange(0, 1.2, 0.05))
    #     plt.ylim(0, 1)
    #     ax.plot(range(len(rank_s)),exp,color='orange',label="experiment result")
    #     ax.plot(range(len(rank_s)),[0.5]*len(exp),label="50%")
    #     # ax.plot(range(len(rank_s)),thm,color='blue',label="theorem result")
    #     ax.set_title("d_"+str(name[0])+",cof_"+str(name[3])+",m_"+str(name[1])+"_with "+str(name[4])+" copies")
    #     ax.set_xlabel("the ranks")
    #     ax.set_ylabel("Success Probability")
    #     ax.legend()
        
    #     plt.savefig(save_dir+"/"+name[2]+"/d_"+str(name[0])+"gate_number_cof"+str(name[3])+",m_"+str(name[1])+",copy_"+str(name[4])+".png")
    
    # grouped = filtered_df.groupby(['d',  'method', 'gate_num_time','copies'])
    
    
    # for name,group in grouped:
        
    #     sub_grouped = group.groupby(['d', 'm', 'method', 'gate_num_time','copies'])
    #     exp_data_s=[]
    #     thm_data_s=[]
    #     m_s=[]
    #     for sub_name,sub_group in sub_grouped:
            
    #         exp=[]
    #         # thm=[]
    #         rank_s=[]
    #         m_s.append(sub_name[1])
    #         for item in sub_group.values:
                
    #             exp.append(np.mean(item[7]["experiment"]))
                
    #             rank_s.append(item[2])
                
                    
    #         exp_data_s.append(exp)
    #         # thm_data_s.append(thm)
        
    
    # fig,ax=plt.subplots()
    # plt.xticks(range(len(rank_s)),labels=rank_s)
    # plt.yticks(np.arange(0, 1.2, 0.05))
    # plt.ylim(0, 1)
    # for exp_data,m in zip(exp_data_s,m_s):
    #     ax.plot(range(len(rank_s)),exp_data,label="m_"+str(m))
        
    #     # ax.plot(range(len(rank_s)),thm_data,label=+"_thm")
    # ax.plot(range(len(rank_s)),[0.5]*len(exp),label="50%")
    # ax.set_title("The comparison in different ranks with "+str(sub_name[4])+" copies")
    # ax.set_xlabel("the ranks")
    # ax.set_ylabel("Success Probability")
    # ax.legend()
    
    # plt.savefig(save_dir+"/"+sub_name[2]+"/d_"+str(sub_name[0])+"gate_number_cof"+str(sub_name[3])+",copy_"+str(sub_name[4])+".png")
    
    ################################ m plot ##############################################################
    # Define the inverse power law model with an asymptote at y = 0.5: y = 0.5 + b * x^(-a)
    def inverse_power_law_with_asymptote(x, a, b,c):
        return c + b * x**(-a)

    # Group data by specified parameters and perform regression analysis for each group
    grouped = pd_data.groupby(['d', 'method', 'gate_num_time', 'copies'])
    for name, group in grouped:
        
        exp = []
        m_s = []
        exp_std = []
        
        m_group = group.groupby('m')
        
        for sub_name, sub_group in m_group:
            
            m_s.append(sub_name)
            take_average_exp = []
            
            for item in sub_group.values:
                take_average_exp.append(item[7]['experiment'])
            
            # Calculate mean and standard deviation for experiment values
            exp.append(np.mean(take_average_exp))
            exp_std.append(np.std(take_average_exp))
        
        ####################################################################################
        fig, ax = plt.subplots()
        plt.yticks(np.arange(0, 1.2, 0.05))
        plt.ylim(0, 1)

        # Plot the 50% line if applicable
        if name[1] in ["special_blended", "special_random", "interweave", "blended_three", "classical_shadow"]:
            ax.plot(m_s, [0.5] * len(exp), label="50%", color='blue')
            plt.ylim(0, 1)

        # Plot the experimental results with error bars
        ax.errorbar(m_s, exp, yerr=exp_std, color='#9ACD32', label="experiment result", capsize=3)

        # Fit and plot the inverse power law regression with asymptote at 0.5
        initial_guesses = [0.5, 0.5,0.5]  # Initial guesses for a and b
        lower_bounds = [0, 0, 0.5]   # a > 0, b > 0, c > 0.5
        upper_bounds = [np.inf, np.inf, np.inf]  # No upper bound constraints
        try:
            params, _ = curve_fit(inverse_power_law_with_asymptote, m_s, exp, p0=initial_guesses, bounds=(lower_bounds, upper_bounds),maxfev=2000)
            a, b ,c= params
            x_fit = np.linspace(min(m_s), max(m_s), 100)
            y_inv_fit = inverse_power_law_with_asymptote(x_fit, a, b,c)
            ax.plot(x_fit, y_inv_fit,color='goldenrod', label=f'Regression Fit')
            
            # Display the inverse power law formula on the plot
            inv_formula_text = f"Inverse Power Law: y = {c:.3f} + {b:.3f} * x^(-{a:.3f})"
            ax.text(0.05, 0.05, inv_formula_text, transform=ax.transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7), ha='left', va='bottom')
            
        except RuntimeError:
            print(f"Could not fit inverse power law for group {name}")

        # Set specific x-ticks to match [4, 8, ..., 32]
        plt.xticks(m_s, labels=m_s)

        # Set title, labels, and legend
        ax.set_title(f"d_{name[0]}, cof_{name[2]}, with {name[3]} copies")
        ax.set_xlabel("The number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend(loc='upper right')

        # Save the plot
        plt.savefig(f"{save_dir}/{name[1]}/d_{name[0]}gate_number_cof{name[2]}, with {name[3]} copies.png")



                    
    # # ####################################### method plot ############################################################   
    filtered_df = pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'classical_shadow') |(pd_data['method'] == 'blended_three') | (pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') | (pd_data['method'] == 'special_random')|(pd_data['method'] == 'optimizing_blended'))]
    filtered_blended=pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'blended_three') | (pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') | (pd_data['method'] == 'optimizing_blended'))]
    grouped = filtered_df.groupby(['d','copies'])
    colors = ['#04D8B2','#008000','#FC5D50','#FFA500','#9ACD32','#000000']
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
                    
                    take_average_exp.append(item[7]['experiment'])
                    # take_average_thm.append(item[7]['theorem'])
                
                exp.append(np.mean(take_average_exp))
            
            method_plot.append(exp)
            method.append(sub_name[0])
    
         
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%")
        plt.ylim(0, 1)
        
        for method_name, plot,color in zip(method,method_plot,colors):
            ax.plot(range(len(m_s)),plot,label=method_name,color=color)
        
        # ax.set_title("the overall comprison")
        ax.set_title("the number of copy= "+str(name[1]))
        # ax.set_title("the comparison in derivations of blended measurement")
        ax.set_xlabel("the number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"d_"+str(name[0])+"blended_derivation_"+"_copies_"+str(name[1])+".png")
    
    # # # ########################################## gate num time plot ####################################################
    
    # ### compare different cof for each interweave and special_blended
    
    # filtered_df = pd_data[(pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')]
    
    # grouped = filtered_df.groupby(['d','copies'])
    
    
    # for name,group in grouped:
    #     # print(group)
        
    #     d_group=group.groupby(['method'])
    #     for sub_name,sub_group in d_group:
           
            
    #         # print()
    #         gate_num_plot=[]
    #         gate_num=[]
    #         gate_num_group=sub_group.groupby(['gate_num_time'])
    #         for sub_sub_name,sub_sub_group in gate_num_group:
    #             # print(sub_sub_group)
    #             # print()
    #             exp=[]
    #             m_s=[]
    #             m_group=sub_sub_group.groupby('m')
                
    #             for sub_sub_sub_name, sub_sub_sub_group in m_group:
                    
    #                 m_s.append(sub_sub_sub_name)
    #                 take_average_exp=[]
    #                 # take_average_thm=[]
    #                 for item in sub_sub_sub_group.values:
                        
    #                     take_average_exp.append(item[7]['experiment'])
    #                     # take_average_thm.append(item[7]['theorem'])
                    
    #                 exp.append(np.mean(take_average_exp))
                
    #             gate_num_plot.append(exp)
    #             gate_num.append(sub_sub_name[0])

            
         
    #         fig,ax=plt.subplots()
    #         plt.xticks(range(len(m_s)),labels=m_s)
    #         plt.yticks(np.arange(0, 1.2, 0.05))
            
            
    #         ax.plot(range(len(m_s)),[0.5]*len(m_s),label="50%")
    #         plt.ylim(0, 1)
            
    #         for gate_name, plot in zip(gate_num,gate_num_plot):
    #             ax.plot(range(len(m_s)),plot,label="cof="+str(gate_name))
            
    #         ax.set_title("d_"+str(name[0])+",copies_"+str(name[1]))
    #         ax.set_xlabel("the number of measurements")
    #         ax.set_ylabel("Success Probability")
    #         ax.legend()
            
    #         plt.savefig(save_dir+"/"+sub_name[0]+"/d_"+str(name[0])+",copies_"+str(name[1])+".png")
    
    # # # ####################################################################################################################
    
    ### compare interweave and special_blended in each Alpha cof
    cof_s=[
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        1.2,
        1.4,
        1.6,
        1.8,
        2,
        2.2,
        2.4]
    for cof in cof_s:
        filtered_df = pd_data[(pd_data["gate_num_time"]==cof) &((pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') )]
    
        grouped = filtered_df.groupby([ 'd','copies'])
        
        
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
                    # take_average_thm=[]
                    for item in sub_sub_group.values:
                        
                        take_average_exp.append(item[7]['experiment'])
                        # take_average_thm.append(item[7]['theorem'])
                    
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
            
            ax.set_title("d_"+str(name[0])+"_cof_"+str(cof)+"_copies_"+str(name[1]))
            ax.set_xlabel("the number of measurements")
            ax.set_ylabel("Success Probability")
            ax.legend()
            
            plt.savefig(save_dir+"/"+"d_"+str(name[0])+"_cof_"+str(cof)+"_copies_"+str(name[1])+".png")
    
    # ##################################################################################################################
    
    ## compare interweave and specail_blended in copies in with different alpha 
    m=12
    
    filtered_df = pd_data[(pd_data["copies"]==1)&(pd_data["m"]==m) &((pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')  )]

    # print(filtered_df)
    grouped = filtered_df.groupby(['d','copies'])
    
    
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
                # take_average_thm=[]
                for item in sub_sub_group.values:
                    
                    take_average_exp.append(item[7]['experiment'])
                    # take_average_thm.append(item[7]['theorem'])
                
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
        
        ax.set_title("m= "+str(m))
        ax.set_xlabel("the different cofs")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"cofs_m_"+str(m)+"_copies_"+str(name[1])+".png")
    
    # #####################################################################################################
    
    # ###compare interweave, special_blended, three-outcome, classical_shadow with different copies
    
    # filtered_df = pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')  | (pd_data['method'] =='blended_three') | (pd_data['method'] =='classical_shadow')| (pd_data['method'] =='special_random'))]

    # # print(filtered_df)
    # grouped = filtered_df.groupby(['d','copies'])
    
    
    # for name,group in grouped:
    #     # print("######################")
    #     method_plot=[]
    #     method=[]
    #     d_group=group.groupby(['method'])
    #     for sub_name,sub_group in d_group:
    #         # print(sub_group) 
    #         # print()
    #         exp=[]
    #         m_s=[]
    #         m_group=sub_group.groupby('m')
            
    #         for sub_sub_name, sub_sub_group in m_group:
                
    #             take_average_exp=[]
    #             take_average_thm=[]
    #             for item in sub_sub_group.values:
                    
    #                 take_average_exp.append(item[7]['experiment'])
    #                 take_average_thm.append(item[7]['theorem'])
                
    #             exp.append(np.mean(take_average_exp))
    #             m_s.append(sub_sub_name)
          
    #         method_plot.append(exp)
        
    #         method.append(sub_name[0])
        
        
    #     fig,ax=plt.subplots()
    #     plt.xticks(range(len(m_s)),labels=m_s)
    #     plt.yticks(np.arange(0, 1.2, 0.05))
        
        
    #     ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%")
    #     plt.ylim(0, 1)
        
    #     for method_name, plot in zip(method,method_plot):
    #         ax.plot(range(len(m_s)),plot,label=method_name)
        
    #     ax.set_title("d_"+str(name[0])+"_copies_"+str(name[1]))
    #     ax.set_xlabel("the different cofs")
    #     ax.set_ylabel("Success Probability")
    #     ax.legend()
        
    #     plt.savefig(save_dir+"/"+"d_m_"+str(name[0])+"_copies_"+str(name[1])+".png")
    ##########################################################################################################################
   
    # filtered_df = pd_data[((pd_data['method'] == 'blended')|(pd_data['method'] == 'random') )]
    # grouped = filtered_df.groupby(['d','method','case'])
    
    # for name,group in grouped:
        
    #     exp=[]
    #     thm=[]
    #     m_s=[]
        
        
    #     m_group=group.groupby('m')
        
    #     for sub_name, sub_group in m_group:
            
    #         m_s.append(sub_name)
    #         take_average_exp=[]
    #         take_average_thm=[]
    #         for item in sub_group.values:
                
    #             take_average_exp.append(item[7]['experiment'])
    #             take_average_thm.append(item[7]['theorem'])
            
    #         exp.append(np.mean(take_average_exp))
    #         thm.append(np.mean(take_average_thm))
        
    #     fig,ax=plt.subplots()
    #     plt.xticks(range(len(m_s)),labels=m_s)
    #     plt.yticks(np.arange(0, 1.2, 0.05))
    #     plt.ylim(0, 1)

    #     ax.plot(range(len(m_s)),thm,color='blue',label="bound")
    #     ax.plot(range(len(m_s)),exp,color='orange',label="experiment result")
        
    #     ax.set_title("d: "+str(name[0])+", method: "+str(name[1])+", case: "+str(name[2]))
    #     ax.set_xlabel("the number of measurements")
    #     if name[2]==1:
    #         ax.set_ylabel("Success Probability")
    #     elif name[2]==2:
    #         ax.set_ylabel("Fail Probability")
    #     ax.legend()
        
    #     plt.savefig(save_dir+"/"+name[1]+"/d_"+str(name[0])+", method_"+str(name[1])+", case_"+str(name[2])+".png")
     
    

