import numpy as np
import random
from scipy.stats import unitary_group
import time
import sys
from tools import projector_html
from scipy.stats import unitary_group
from generate_vector import generate_first_vector


###############################################################################################################

import matplotlib.pyplot as plt
from scipy.stats import truncnorm


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
    
    # Generate low-probability POVMs
    while len(povm) < number_of_low:
        
        # epson_l=random.uniform(0.03,low_pro) 
        # #generate the epson vector
        # epson_vector_l=np.hstack(((np.sqrt(1-epson_l)),np.zeros(d-rank-1),(np.sqrt(epson_l)),np.zeros(rank-1)))
        p=distribution(0.00001,low_pro)
        epson_vector_l=generate_first_vector(d,p)
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_l)
        
        #rotating
        temp=U.T.conj()@projector@U
       
        print(f'probability: {np.real(np.trace(temp @ roh))}')
        #append povm to the set
        povm.append(temp)
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()

    # Generate high-probability POVMs
    while len(povm) < m:
        # epson_h=random.uniform(high_pro,1)
        
        # #generate the epson vector
        # epson_vector_h=np.hstack(((np.sqrt(1-epson_h)),np.zeros(d-rank-1),(np.sqrt(epson_h)),np.zeros(rank-1)))
        p=distribution(high_pro,1)
        epson_vector_h=generate_first_vector(d,p)
         
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_h)
        
        #rotating
        temp=U.T.conj()@projector@U
        
        print(f'probability: {np.real(np.trace(temp @ roh))}')
        #append povm to the set
        povm.append(temp)
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()
    
    return povm

    
def yieldRandomUnitary(d,epson_vector):
    
    A=unitary_group.rvs(d)
    A[:, 0] = epson_vector
    U, R = np.linalg.qr(A)

    return U



def distribution(l,h):
    # Parameters for the normal distribution
    mu = (l+h)/2     # Mean (chosen to be within the truncation range)
    sigma = 0.02  # Standard deviation

    # Truncation boundaries
    a, b = l,h

    # Convert to the standard normal form for truncnorm
    a_standard = (a - mu) / sigma
    b_standard = (b - mu) / sigma

    # Generate samples from the truncated normal distribution
    sample = truncnorm.rvs(a_standard, b_standard, loc=mu, scale=sigma, size=1)

    return sample
    
    
    # # Plot the histogram of the samples
    # plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

    # # Plot the truncated normal distribution for reference
    # x = np.linspace(0, 0.1, 1000)
    # pdf = truncnorm.pdf(x, a_standard, b_standard, loc=mu, scale=sigma)
    # plt.plot(x, pdf, 'r-', lw=2)

    # plt.title('Truncated Normal Distribution (0 ~ 0.1)')
    # plt.show()
