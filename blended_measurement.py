import numpy as np
from scipy.linalg import sqrtm


def blended_measurement(set,d,m):
    E=[]
    sum_set=np.zeros((d, d), dtype=np.complex128)
    
    for item in set:
        sum_set+=item
    
    identity=np.eye(d)
    E_0=sqrtm(identity-sum_set/m)
    E_0=E_0.astype('complex128')
    E.append(E_0)
    for item in set:
        temp=sqrtm(item/m)
        temp=temp.astype('complex128')
        E.append(temp)
   
    return E


# def optimizing_blended_measurement(set,d,m):
#     E=[]
#     sum_set=np.zeros((d, d), dtype=np.complex128)
    
#     optimized=[]
    
#     for item in set:
#         eigenValue, eigenvector= np.linalg.eig(item)
#         maxEigenValue=max(abs(eigenValue)/m)
#         print(maxEigenValue)
#         optimizingFactor=1/maxEigenValue
#         print(optimizingFactor)
#         opt=optimizingFactor*(item/m)
#         optimized.append(opt)
#         print(abs(opt))
        
#     identity=np.eye(d)
    
#     for item in optimized:
#         sum_set+=item    
#     print(abs(sum_set))
#     E_0=identity-sum_set
#     E_0=E_0.astype('complex128')
#     eigenValue, eigenvector= np.linalg.eig(E_0)
#     # print(abs(eigenValue))

#     E.append(E_0)
#     for item in optimized:
#         temp=sqrtm(item)
#         temp=temp.astype('complex128')
        
    
#         E.append(temp)
    
#     return E
    

def blended_measurement_inverse(set,d,m):
    E=[]
    identity=np.eye(d)
    
    
    sum_set=np.zeros((d, d), dtype=np.complex128)
    for item in set:
        sum_set+=(identity-item)
    
    E_0=sqrtm(identity-(sum_set/m))
    E_0=E_0.astype('complex128')
    E.append(E_0)
    for item in set:
        
        temp=sqrtm((identity-item)/m)
        temp=temp.astype('complex128')
        E.append(temp)
    
   
    return E


def optimizing_blended_measurement(set,d,m):
    E=[]
    sum_set=np.zeros((d, d), dtype=np.complex128)
    
    for item in set:
        eigenValue, eigenvector= np.linalg.eig(item)
        print(f'eigenValue: {abs(eigenValue)}')
        sum_set+=item
    eigenValue, eigenvector= np.linalg.eig(sum_set)
    maxEigenValue=max(abs(eigenValue)/m)
    print(f'the max in the eigenValue: {maxEigenValue}')
    optimizingFactor=1/maxEigenValue
    print(f'the factor: {optimizingFactor}')
    identity=np.eye(d)
    E_0=sqrtm(identity-(sum_set/m)*optimizingFactor)
    E_0=E_0.astype('complex128')
    eigenValue, eigenvector= np.linalg.eig(E_0)
    # print(abs(eigenValue))
    
    E.append(E_0)
    for item in set:
        temp=sqrtm((item/m)*optimizingFactor)
        temp=temp.astype('complex128')
        eigenValue, eigenvector= np.linalg.eig(temp)
    
        E.append(temp)
    
    return E