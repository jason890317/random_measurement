import numpy as np
import random
from scipy.stats import unitary_group
import time
import sys
import os

from tools import generate_rank_n_projector, generate_random_statevector,show_probability_povm,projector_html
def generate_normalized_psd_matrix(m,d,high=True):
    """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
    np.random.seed(int(time.time()))
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    A = A + A.conj().T 
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    normalized_eigenvalues=[]
    if high:
        normalized_eigenvalues=[random.uniform(0.7, 1) for _ in range(d)]
    else:
        normalized_eigenvalues=[random.uniform(0, 0.2/m) for _ in range(d)]
        # print(len(normalized_eigenvalues))
    return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T
    
        
def generate_povm_set_case_1(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m-1)]
    povm_elements.append(generate_normalized_psd_matrix(m,d,True))

    return povm_elements


def generate_povm_set_case_2(d, m):
    """Generate a POVM set with specified properties."""
    povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m)]
    
    return povm_elements

# def generate_povm_by_unitary_case_1(d,m,rank_h,rank_l,roh):
    
#     povm=[]
#     seed=int(time.time())
#     len_povm=0
    
#     projector_high=generate_rank_n_projector(rank_h,d)
#     projector_low=generate_rank_n_projector(rank_l,d)
    
#     while len(povm)!=m-1:
#         np.random.seed(seed)
#         U=unitary_group.rvs(d)
#         temp=U@projector_low@U.T.conj()
#         if 0<np.trace(temp@roh)<0.1:
#             povm.append(temp)
#         seed+=1
#         if len_povm!=len(povm):
#             sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
#             sys.stdout.flush()
            
#         len_povm=len(povm)
#     while len(povm)!=m:
#         np.random.seed(seed)
#         U=unitary_group.rvs(d)
#         temp=U@projector_high@U.T.conj()
#         if 1>np.trace(temp@roh)>0.7:
#             povm.append(temp)
#         seed+=1
#         if len_povm!=len(povm):
#             sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
#             sys.stdout.flush()
            
#         len_povm=len(povm)
#     print()
#     return povm

# def generate_povm_by_unitary_case_2(d,m,rank,roh):
   
#     povm=[]
#     seed=int(time.time())
#     len_povm=0
    
#     projector=generate_rank_n_projector(rank,d)
    
#     while len(povm)!=m:
#         np.random.seed(seed)
#         U=unitary_group.rvs(d)
#         temp=U@projector@U.T.conj()
#         if 0<np.trace(temp@roh)<0.3/m:
#             povm.append(temp)
#         seed+=1
#         if len_povm!=len(povm):
#             sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
#             sys.stdout.flush()
            
#         len_povm=len(povm)
    
#     print()
#     return povm


def generate_povm_epson_case_1(d,m,rank,pro_h,pro_l,roh):
     
    povm=[]
    
    len_povm=0
    projector=np.diag(np.hstack([np.zeros(d-rank), np.ones(rank)]))
    
    dir_name="d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(1)+"_projector.html"
    
    
    while len(povm)!=m-1:
        
        epson_l=random.uniform(0.03,pro_l) #
        # print("epson: "+str(epson_l))
        epson_vector_l=np.hstack(((np.sqrt(1-epson_l)),np.zeros(d-rank-1),(np.sqrt(epson_l)),np.zeros(rank-1)))
        # print(epson_vector_l)
        # epson_vector_l /=np.linalg.norm(epson_vector_l)
        real_part = np.random.rand(d, d)
        imaginary_part = np.random.rand(d, d)
        A= real_part + 1j * imaginary_part
        A[:, 0] = epson_vector_l
        # print(f'ever A: \n{A}')
        U, R = np.linalg.qr(A)
        # print(f'every U: \n{U}')
        # print(f'P: \n{projector}')
        temp=U.T.conj()@projector@U
        # print(np.trace(temp@roh))
        
        projector_html(dir_name,temp,4)
            

        povm.append(temp)
        
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    while len(povm)!=m:
        
        epson_h=random.uniform(pro_h,1)
        epson_vector_h=np.hstack(((np.sqrt(1-epson_h)),np.zeros(d-rank-1),(np.sqrt(epson_h)),np.zeros(rank-1)))
        epson_vector_h /=np.linalg.norm(epson_vector_h)
        real_part = np.random.rand(d, d)
        imaginary_part = np.random.rand(d, d)
        A= real_part + 1j * imaginary_part
        A[:, 0] = epson_vector_h
        # print(f'ever A: \n{A}')
        U, R = np.linalg.qr(A)
        # print(f'every U: \n{U}')
        # print(f'P: \n{projector}')
        temp=U.T.conj()@projector@U
        # print(f'pro: \n{np.trace(temp@roh)}')

        projector_html(dir_name,temp,4)
            
                
        povm.append(temp)
        
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    print()
    # show_probability_povm(povm,roh,print_pro=True)
    return povm

def generate_povm_epson_case_2(d,m,rank,pro_l,roh):
   
    povm=[]
    
    len_povm=0
    
    dir_name="d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(2)+"_projector.html"

    projector=np.diag(np.hstack([np.zeros(d-rank), np.ones(rank)]))
    
    while len(povm)!=m:
        epson=random.uniform(0,pro_l)
        epson_vector=np.hstack(((np.sqrt(1-epson)),np.zeros(d-rank-1),(np.sqrt(epson)),np.zeros(rank-1)))
        epson_vector /=np.linalg.norm(epson_vector)
        real_part = np.random.rand(d, d)
        imaginary_part = np.random.rand(d, d)
        A= real_part + 1j * imaginary_part
        A[:, 0] = epson_vector
        U, R = np.linalg.qr(A)
        
        temp=U.T.conj()@projector@U
        # print(f'projector: {temp}')
        povm.append(temp)
        
        
        projector_html(dir_name,temp,4)
        
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
            
        len_povm=len(povm)
    
    print()
    return povm
    