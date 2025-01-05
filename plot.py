import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import sys
from scipy.optimize import curve_fit
from guessing import generate_list, generate_success_pro

def x_axis_rank_plot(pd_data,save_dir):
    filtered_df = pd_data[(pd_data["gate_num_time"] == 1) & (pd_data["copies"] == 20)]
    grouped = filtered_df.groupby(['dimension', 'm', 'method', 'gate_num_time', 'copies'])

    for name, group in grouped:
        exp = [np.mean(item[7]["experiment"]) for item in group.values]
        rank_s = [item[2] for item in group.values]

        fig, ax = plt.subplots()
        ax.plot(range(len(rank_s)), exp, color='orange', label="experiment result")
        ax.plot(range(len(rank_s)), [0.5] * len(exp), label="50%",linestyle='-.')
        ax.set_title(f"d_{name[0]}, cof_{name[3]}, m_{name[1]} with {name[4]} copies")
        ax.set_xlabel("the ranks")
        ax.set_ylabel("Success Probability")
        ax.legend()
        plt.xticks(range(len(rank_s)), labels=rank_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        plt.ylim(0, 1)
        plt.savefig(f"{save_dir}/{name[2]}/d_{name[0]}gate_number_cof{name[3]},m_{name[1]},copy_{name[4]}.png")

    grouped = filtered_df.groupby(['dimension', 'method', 'gate_num_time', 'copies'])

    for name, group in grouped:
        sub_grouped = group.groupby(['dimension', 'm', 'method', 'gate_num_time', 'copies'])
        exp_data_s = []
        m_s = []

        for sub_name, sub_group in sub_grouped:
            exp = [np.mean(item[7]["experiment"]) for item in sub_group.values]
            rank_s = [item[2] for item in sub_group.values]
            exp_data_s.append(exp)
            m_s.append(sub_name[1])

        fig, ax = plt.subplots()
        for exp_data, m in zip(exp_data_s, m_s):
            ax.plot(range(len(rank_s)), exp_data, label=f"m_{m}")
        ax.plot(range(len(rank_s)), [0.5] * len(exp), label="50%",linestyle='-.')
        ax.set_title(f"The comparison in different ranks with {sub_name[4]} copies")
        ax.set_xlabel("the ranks")
        ax.set_ylabel("Success Probability")
        ax.legend()
        plt.xticks(range(len(rank_s)), labels=rank_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        plt.ylim(0, 1)
        plt.savefig(f"{save_dir}/{sub_name[2]}/d_{sub_name[0]}gate_number_cof{sub_name[3]},copy_{sub_name[4]}.png")


def x_axis_number_of_measurement_plot(pd_data,save_dir):
    def inverse_power_law_with_asymptote(x, a, b, c):
        return c + b * x**(-a)

    grouped = pd_data.groupby(['dimension', 'method', 'gate_num_time', 'copies'])
    for name, group in grouped:
        exp, exp_std, m_s = [], [], []
        for m, sub_group in group.groupby('m'):
            m_s.append(m)
            exp_values = [item[7]['experiment'] for item in sub_group.values]
            exp.append(np.mean(exp_values))
            exp_std.append(np.std(exp_values))

        fig, ax = plt.subplots()
        ax.errorbar(m_s, exp, yerr=exp_std, color='#9ACD32', label="experiment result", capsize=3)
        if name[1] in ["special_blended", "special_random", "interweave", "blended_three", "classical_shadow"]:
            ax.plot(m_s, [0.5] * len(exp), label="50%", color='blue',linestyle='-.')

        try:
            params, _ = curve_fit(inverse_power_law_with_asymptote, m_s, exp, p0=[0.5, 0.5, 0.5], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=2000)
            x_fit = np.linspace(min(m_s), max(m_s), 100)
            ax.plot(x_fit, inverse_power_law_with_asymptote(x_fit, *params), color='goldenrod', label='Regression Fit')
            ax.text(0.05, 0.05, f"y = {params[2]:.3f} + {params[1]:.3f} * x^(-{params[0]:.3f})", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        except RuntimeError:
            print(f"Could not fit inverse power law for group {name}")

        ax.set_title(f"d_{name[0]}, cof_{name[2]}, with {name[3]} copies")
        # ax.set_title(f"The Overall Comparison")
        ax.set_xlabel("The number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend(loc='upper right')
        plt.xticks(m_s, labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        plt.ylim(0, 1)
        plt.savefig(f"{save_dir}/{name[1]}/d_{name[0]}gate_number_cof{name[2]}, with {name[3]} copies.png")


def across_all_methods_plot(pd_data,save_dir):
    filtered_df = pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'classical_shadow') |(pd_data['method'] == 'blended_three') | (pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') | (pd_data['method'] == 'special_random')|(pd_data['method'] == 'optimizing_blended'))]
    filtered_blended=pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'blended_three') | (pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave') | (pd_data['method'] == 'optimizing_blended'))]
    grouped = filtered_blended.groupby(['dimension','copies'])
    colors = ['#04D8B2','#008000','#FC5D50','#FFA500','#9ACD32','#000000','#FF0000']
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
        # method.append('randomly_guessing')
        # random_guess=[]
        # for m in m_s:
        #     generate_success_pro(m,250,random_guess)
        # method_plot.append(random_guess)
        
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%",linestyle='-.')
        
        plt.ylim(0, 1)
        
        for method_name, plot,color in zip(method,method_plot,colors):
            if method_name=='randomly_guessing':
                ax.plot(range(len(m_s)), plot, label=method_name, color=color, linestyle='--')
            else:
                ax.plot(range(len(m_s)),plot,label=method_name,color=color)
        
        # ax.set_title("the overall comprison")
        # ax.set_title("the number of copy= "+str(name[1]))
        ax.set_title("the comparison in derivations of blended measurement")
        ax.set_xlabel("the number of measurements")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"d_"+str(name[0])+"blended_derivation_"+"_copies_"+str(name[1])+".png")

def blended_deritative_measurement_impletement_number_plot(pd_data,save_dir):
    ### compare different cof for each interweave and special_blended
    
    filtered_df = pd_data[(pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')]
    
    grouped = filtered_df.groupby(['dimension','copies'])
    
    
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
                    # take_average_thm=[]
                    for item in sub_sub_sub_group.values:
                        
                        take_average_exp.append(item[7]['experiment'])
                        # take_average_thm.append(item[7]['theorem'])
                    
                    exp.append(np.mean(take_average_exp))
                
                gate_num_plot.append(exp)
                gate_num.append(sub_sub_name[0])

            
         
            fig,ax=plt.subplots()
            plt.xticks(range(len(m_s)),labels=m_s)
            plt.yticks(np.arange(0, 1.2, 0.05))
            
            
            ax.plot(range(len(m_s)),[0.5]*len(m_s),label="50%",linestyle='-.')
            plt.ylim(0, 1)
            
            for gate_name, plot in zip(gate_num,gate_num_plot):
                ax.plot(range(len(m_s)),plot,label="cof="+str(gate_name))
            
            ax.set_title("d_"+str(name[0])+",copies_"+str(name[1]))
            ax.set_xlabel("the number of measurements")
            ax.set_ylabel("Success Probability")
            ax.legend()
            
            plt.savefig(save_dir+"/"+sub_name[0]+"/d_"+str(name[0])+",copies_"+str(name[1])+".png")

def blended_deritatives_cof_plot(pd_data,save_dir):
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
    
        grouped = filtered_df.groupby([ 'dimension','copies'])
        
        
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
            
            
            ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%",linestyle='-.')
            plt.ylim(0, 1)
            
            for method_name, plot in zip(method,method_plot):
                ax.plot(range(len(m_s)),plot,label=method_name)
            
            ax.set_title("d_"+str(name[0])+"_cof_"+str(cof)+"_copies_"+str(name[1]))
            ax.set_xlabel("the number of measurements")
            ax.set_ylabel("Success Probability")
            ax.legend()
            
            plt.savefig(save_dir+"/"+"d_"+str(name[0])+"_cof_"+str(cof)+"_copies_"+str(name[1])+".png")
    


def single_measurement_comparison_across_cof_plot(pd_data,m,save_dir):
    ## compare interweave and specail_blended in copies in with different alpha 
    
    
    filtered_df = pd_data[(pd_data["copies"]==1)&(pd_data["m"]==m) &((pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')  )]

    # print(filtered_df)
    grouped = filtered_df.groupby(['dimension','copies'])
    
    
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
        
        
        ax.plot(range(len(cof_s)),[0.5]*len(exp),label="50%",linestyle='-.')
        plt.ylim(0, 1)
        
        for method_name, plot in zip(method,method_plot):
            ax.plot(range(len(cof_s)),plot,label=method_name)
        
        ax.set_title("m= "+str(m))
        ax.set_xlabel("the different cofs")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"cofs_m_"+str(m)+"_copies_"+str(name[1])+".png")


def across_copies_plot(pd_data,save_dir):
    ###compare interweave, special_blended, three-outcome, classical_shadow with different copies
    
    filtered_df = pd_data[(pd_data["gate_num_time"]==1) &((pd_data['method'] == 'optimizing_blended')|(pd_data['method'] == 'special_blended') | (pd_data['method'] =='interweave')  | (pd_data['method'] =='blended_three') | (pd_data['method'] =='classical_shadow')| (pd_data['method'] =='special_random'))]

    # print(filtered_df)
    grouped = filtered_df.groupby(['dimension','copies'])
    
    
    for name,group in grouped:
        # print("######################")
        method_plot=[]
        method=[]
        d_group=group.groupby(['method'])
        for sub_name,sub_group in d_group:
            # print(sub_group) 
            # print()
            exp=[]
            m_s=[]
            m_group=sub_group.groupby('m')
            
            for sub_sub_name, sub_sub_group in m_group:
                
                take_average_exp=[]
                take_average_thm=[]
                for item in sub_sub_group.values:
                    
                    take_average_exp.append(item[7]['experiment'])
                    take_average_thm.append(item[7]['theorem'])
                
                exp.append(np.mean(take_average_exp))
                m_s.append(sub_sub_name)
          
            method_plot.append(exp)
        
            method.append(sub_name[0])
        
        
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        
        
        ax.plot(range(len(m_s)),[0.5]*len(exp),label="50%",linestyle='-.')
        plt.ylim(0, 1)
        
        for method_name, plot in zip(method,method_plot):
            ax.plot(range(len(m_s)),plot,label=method_name)
        
        ax.set_title("d_"+str(name[0])+"_copies_"+str(name[1]))
        ax.set_xlabel("the different cofs")
        ax.set_ylabel("Success Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+"d_m_"+str(name[0])+"_copies_"+str(name[1])+".png")

def event_finding_plot(pd_data,save_dir):
    filtered_df = pd_data[((pd_data['method'] == 'blended')|(pd_data['method'] == 'random') )]
    grouped = filtered_df.groupby(['dimension','method','case'])
    
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
                
                take_average_exp.append(item[7]['experiment'])
                take_average_thm.append(item[7]['theorem'])
            
            exp.append(np.mean(take_average_exp))
            thm.append(np.mean(take_average_thm))
        
        fig,ax=plt.subplots()
        plt.xticks(range(len(m_s)),labels=m_s)
        plt.yticks(np.arange(0, 1.2, 0.05))
        plt.ylim(0, 1)

        ax.plot(range(len(m_s)),thm,color='blue',label="bound")
        ax.plot(range(len(m_s)),exp,color='orange',label="experiment result")
        
        ax.set_title("d: "+str(name[0])+", method: "+str(name[1])+", case: "+str(name[2]))
        ax.set_xlabel("the number of measurements")
        if name[2]==1:
            ax.set_ylabel("Success Probability")
        elif name[2]==2:
            ax.set_ylabel("Fail Probability")
        ax.legend()
        
        plt.savefig(save_dir+"/"+name[1]+"/d_"+str(name[0])+", method_"+str(name[1])+", case_"+str(name[2])+".png")

def main():
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
    
    x_axis_rank_plot(pd_data,save_dir)
    x_axis_number_of_measurement_plot(pd_data,save_dir)
    across_all_methods_plot(pd_data,save_dir)
    blended_deritative_measurement_impletement_number_plot(pd_data,save_dir)
    blended_deritatives_cof_plot(pd_data,save_dir)
    for m in range(4,25,4):
        single_measurement_comparison_across_cof_plot(pd_data,m,save_dir)
    across_copies_plot(pd_data,save_dir)
    
    
    
if __name__ == "__main__":
    main()
