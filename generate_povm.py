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
        
        #generate the special vector
        p=distribution(0.003,low_pro)
        epson_vector_l=generate_first_vector(d,p)
        
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_l)
        
        #rotating
        temp=U.T.conj()@projector@U
       
        #append povm to the set
        if np.allclose(temp@temp,temp,atol=(1e-14)):
            povm.append(temp)
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()

    # Generate high-probability POVMs
    while len(povm) < m:
        
        #generate the special vector
        p=distribution(high_pro,1)
        epson_vector_h=generate_first_vector(d,p)
         
        #generate random unitary to rotate the projector
        U=yieldRandomUnitary(d,epson_vector_h)
        
        #rotating
        temp=U.T.conj()@projector@U
        
        #append povm to the set
        if np.allclose(temp@temp,temp,atol=(1e-14)):
            povm.append(temp)
        
        sys.stdout.write(f"\rpovm : {len(povm)}/{m}")
        sys.stdout.flush()
    
    return povm

    
def yieldRandomUnitary(d,epson_vector):
    
    A=unitary_group.rvs(d)
    A[:, 0] = epson_vector
    U, R = np.linalg.qr(A)

    return U




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import rv_continuous

def distribution(a,b):
# Define the interval [a, b]

    # Define a custom PDF function, for example, a quadratic function
    def normal_pdf(x, mu=0.5, sigma=0.1):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Normalize the PDF over the interval [a, b]
    # the integration along the interval from a to b   
    normalization_constant = quad(normal_pdf, a, b)[0]

    def normalized_custom_pdf(x):
        return normal_pdf(x) / normalization_constant

    class custom_distribution(rv_continuous):
        def _pdf(self, x):
            return normalized_custom_pdf(x)

    # Create an instance of the custom distribution
    custom_dist = custom_distribution(a=a, b=b, name='custom')

    # Generate samples
    sample = custom_dist.rvs(size=1)
    
    return sample