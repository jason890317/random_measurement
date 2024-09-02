import numpy as np
import random
from scipy.stats import unitary_group
import time
import sys
from tools import projector_html
from scipy.stats import unitary_group



###############################################################################################################
import numpy as np
from scipy.stats import unitary_group
from joblib import Parallel, delayed
import sys

############################################## generate povm by rotating projector #####################################
def generate_povm_general(case, d, m, rank, case_1_h, case_1_l, case_2_l, roh, batch_size=1000, n_jobs=-1,perturbation=1e-3):
    
    # Determine the number of high and low probability matrices
    if case == 1:
        number_of_high = 1
        high_pro = case_1_h
        low_pro = case_1_l
    elif case == 3:
        number_of_high = int(m / 2)
        high_pro = case_1_h
        low_pro = case_1_l
    elif case == 2:
        number_of_high = 0
        high_pro = 1
        low_pro = case_2_l
        
    number_of_low = m - number_of_high
    povm = []
    projector = np.diag(np.hstack([np.zeros(d - rank), np.ones(rank)]))
    # zero_space_projector = np.eye(d) - projector
    # projector += perturbation * np.random.randn(d, d)
    # projector = (projector + projector.T.conj()) / 2  # Ensure it's still Hermitian
    # Helper function to check low-probability condition
    def check_unitary_low(seed):
        rng = np.random.default_rng(seed)
        U = unitary_group.rvs(d, random_state=rng)
        
        trace_value = np.real(np.trace(U.T.conj() @ projector @ U @ roh))
        if trace_value < low_pro:
            return U.T.conj() @ projector @ U, trace_value
        return None, None

    # Helper function to check high-probability condition
    def check_unitary_high(seed):
        rng = np.random.default_rng(seed)
        U = unitary_group.rvs(d, random_state=rng)
        trace_value = np.real(np.trace(U.T.conj() @ projector @ U @ roh))
        if trace_value > high_pro:
            return U.T.conj() @ projector @ U, trace_value
        return None, None

    # Generate low-probability POVMs
    while len(povm) < number_of_low:
        seeds = np.random.randint(0, np.iinfo(np.int32).max, size=batch_size)
        results = Parallel(n_jobs=n_jobs)(delayed(check_unitary_low)(seed) for seed in seeds)
        
        for result, trace_value in results:
            if result is not None:
                povm.append(result)
                print(f"\npro: {abs(trace_value)}")
            if len(povm) == number_of_low:
                break
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()

    # Generate high-probability POVMs
    while len(povm) < m:
        seeds = np.random.randint(0, np.iinfo(np.int32).max, size=batch_size)
        results = Parallel(n_jobs=n_jobs)(delayed(check_unitary_high)(seed) for seed in seeds)
        
        for result, trace_value in results:
            if result is not None:
                povm.append(result)
                print(f"\npro: {abs(trace_value)}")
            if len(povm) == m:
                break
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()
    
    return povm

    
def yieldRandomUnitary(d,epson_vector):
    
    A=unitary_group.rvs(d)
    A[:, 0] = epson_vector
    U, R = np.linalg.qr(A)

    return U



def generate_povm_epson_case_1(d,m,rank,pro_h,pro_l,roh):
     
    povm=[]
    
    len_povm=0
    
    projector=np.diag(np.hstack([np.zeros(d-rank), np.ones(rank)]))
    
    
    ##################################### projector html ########################################
    # dir_name="d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(1)+"_projector.html"
    
    
    while len(povm)!=m-1: # generate the first m-1 povm with low accepting probability
        
        #choose the epson (accepting probability)
        epson_l=random.uniform(0.03,pro_l) 
        
        #generate the epson vector
        epson_vector_l=np.hstack(((np.sqrt(1-epson_l)),np.zeros(d-rank-1),(np.sqrt(epson_l)),np.zeros(rank-1)))
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_l)
        
        #rotating
        temp=U.T.conj()@projector@U
       
        
        ##################################### projector html ########################################
        # projector_html(dir_name,temp,top_num)
            
        #append povm to the set
        povm.append(temp)
    
        #display the progress bar
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
        
        #update the number of povms in the set
        len_povm=len(povm)
        
    while len(povm)!=m: # generate the hgih accepting probability povm
        
        #choose the epson (accepting probability)
        epson_h=random.uniform(pro_h,1)
        
        #generate the epson vector
        epson_vector_h=np.hstack(((np.sqrt(1-epson_h)),np.zeros(d-rank-1),(np.sqrt(epson_h)),np.zeros(rank-1)))
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_h)
        
        #rotating
        temp=U.T.conj()@projector@U
        

        ##################################### projector html ########################################
        # projector_html(dir_name,temp,top_num)
            
        #append povm to the set
        povm.append(temp)
        
        #display the progress bar
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
        
        #update the number of povms in the set
        len_povm=len(povm)
    
    #display the progress bar
    print()
    
    return povm

def generate_povm_epson_case_2(d,m,rank,pro_l,roh):
   
    povm=[]
    
    len_povm=0
    
    ##################################### projector html ########################################
    # dir_name="d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(2)+"_projector.html"

    projector=np.diag(np.hstack([np.zeros(d-rank), np.ones(rank)]))
    
    while len(povm)!=m:
        
        #choose the epson (accepting probability)
        epson=random.uniform(0,pro_l)
        # epson=random.uniform(0,1)
        #generate the epson vector
        epson_vector=np.hstack(((np.sqrt(1-epson)),np.zeros(d-rank-1),(np.sqrt(epson)),np.zeros(rank-1)))
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector)
        
        #rotating
        temp=U.T.conj()@projector@U
        
        #append povm to the set
        povm.append(temp)
        
        ##################################### projector html ########################################
        # projector_html(dir_name,temp,top_num)
        
        #display the progress bar
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
        
        #update the number of povms in the set
        len_povm=len(povm)
    
    #display the progress bar
    print()
    return povm


def generate_povm_epson_case_special(d,m,rank,pro_h,pro_l,roh):
     
    povm=[]
    
    len_povm=0
    
    projector=np.diag(np.hstack([np.zeros(d-rank), np.ones(rank)]))
    
    
    ##################################### projector html ########################################
    # dir_name="d_"+str(d)+"_m_"+str(m)+"_r_"+str(rank)+"_case_"+str(1)+"_projector.html"
    
    
    while len(povm)!=int(m/2): # generate the first m-1 povm with low accepting probability
        
        #choose the epson (accepting probability)
        epson_l=random.uniform(0.01,pro_l) 
        
        #generate the epson vector
        epson_vector_l=np.hstack(((np.sqrt(1-epson_l)),np.zeros(d-rank-1),(np.sqrt(epson_l)),np.zeros(rank-1)))
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_l)
        
        #rotating
        temp=U.T.conj()@projector@U
       
        
        ##################################### projector html ########################################
        # projector_html(dir_name,temp,top_num)
            
        #append povm to the set
        povm.append(temp)
    
        #display the progress bar
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
        
        #update the number of povms in the set
        len_povm=len(povm)
        
    while len(povm)!=m: # generate the hgih accepting probability povm
        
        #choose the epson (accepting probability)
        epson_h=random.uniform(pro_h,1)
        
        #generate the epson vector
        epson_vector_h=np.hstack(((np.sqrt(1-epson_h)),np.zeros(d-rank-1),(np.sqrt(epson_h)),np.zeros(rank-1)))
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_h)
        
        #rotating
        temp=U.T.conj()@projector@U
        

        ##################################### projector html ########################################
        # projector_html(dir_name,temp,top_num)
            
        #append povm to the set
        povm.append(temp)
        
        #display the progress bar
        if len_povm!=len(povm):
            sys.stdout.write(f"\rpovm : "+str(len_povm+1)+"/"+str(m))
            sys.stdout.flush()
        
        #update the number of povms in the set
        len_povm=len(povm)
    
    #display the progress bar
    print()
    
    return povm

############################### generate povm by manipulating the eigenvalues ########################################

# def generate_normalized_psd_matrix(m,d,high=True):
#     """Generate a normalized PSD matrix with eigenvalues between 0 and 1."""
#     np.random.seed(int(time.time()))
#     A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
#     A = A + A.conj().T 
#     eigenvalues, eigenvectors = np.linalg.eigh(A)
#     normalized_eigenvalues=[]
#     if high:
#         normalized_eigenvalues=[random.uniform(0.7, 1) for _ in range(d)]
#     else:
#         normalized_eigenvalues=[random.uniform(0, 0.2/m) for _ in range(d)]
#         # print(len(normalized_eigenvalues))
#     return eigenvectors @ np.diag(normalized_eigenvalues) @ eigenvectors.conj().T
    
        
# def generate_povm_set_case_1(d, m):
#     """Generate a POVM set with specified properties."""
#     povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m-1)]
#     povm_elements.append(generate_normalized_psd_matrix(m,d,True))

#     return povm_elements


# def generate_povm_set_case_2(d, m):
#     """Generate a POVM set with specified properties."""
#     povm_elements = [generate_normalized_psd_matrix(m,d,False) for _ in range(m)]
    
#     return povm_elements

########################################################################################################################