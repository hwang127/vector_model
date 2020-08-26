#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import linalg
from numpy import linalg as LA
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv,pinv,det,cholesky
def generate_ARMA_sample(ar,ma,T,sig):
    #np.random.seed(12345)
    arparams = np.array(ar)
    maparams = np.array(ma)
    ar = np.r_[1, -arparams] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    y = sm.tsa.arma_generate_sample(ar, ma, T,sig)
    return y
def generate_test_data(N,T,sigma,prt=False): #return data matrix,loadings matrix, factor matrix
    coeff=[[0.8],[0,0,0.9]]
    err_sig=[0.6,0.43]
    f=np.array([generate_ARMA_sample(coeff[i],[],T,err_sig[i]) for i in range(2)])
    risk=np.random.normal(size=[1,T])
    factor=np.vstack([f,risk])
    #sort_index = np.argsort(-f.std(axis=1))
    #f=f[sort_index]
    if prt:
        print('off_diagonal sum=',np.sum(np.abs(f@f.T/T))-np.trace(f@f.T/T))
    #calcualte loading matrix as singular vectors of a random matrix
    
    Gamma=np.random.uniform(-1,1,(N,factor.shape[0]))
    noise=np.random.normal(scale=sigma,size=[N,T])
    X=Gamma@factor+noise

    return X,Gamma,factor,noise

def generate_hetero_test_data(N,T,sigma,prt=False): #return data matrix,loadings matrix, factor matrix
    coeff=[[0.8],[0,0,0.9]]
    err_sig=[0.6,0.43]
    f=np.array([generate_ARMA_sample(coeff[i],[],T,err_sig[i]) for i in range(2)])
    risk=np.random.normal(size=[1,T])
    factor=np.vstack([f,risk])
    if prt:
        print('off_diagonal sum=',np.sum(np.abs(f@f.T/T))-np.trace(f@f.T/T))
    loading=np.random.uniform(-1,1,(N,factor.shape[0]))
    noise=np.diag(np.random.normal(scale=0.5,size=N)+1)@np.random.normal(scale=sigma,size=[N,T])
    X=loading@factor+noise
    return X,loading,factor,noise
def select_r(eigen_Values):
    limit=len(eigen_Values)//2+1
    e_values=eigen_Values[:limit]
    ratios=np.roll(e_values,-1)[:-1]/e_values[:-1]
    #print(ratios)
    return np.argmin(ratios)+1
def calculate_pca(data,r=0):
    U, s, Vh = linalg.svd(data)
    if r==0:
        r=max(select_r(s**2),2)
    Gamma=U[:,:r]
    F=np.dot(Gamma.T,data)
    return Gamma,F
def hetero_pca(data,r,limit=1e-10,max_iter=100):
    Cov=data@data.T
    Cov=Cov-np.diag(Cov.diagonal())
    U,s,Vh=linalg.svd(Cov)
    if r==0:
        r=max(select_r(s),2)
    for i in range(max_iter):
        off_diag=Cov-np.diag(Cov.diagonal())
        old_cov=Cov
        U,s,Vh=linalg.svd(Cov)
        Cov=U[:,:r]@np.diag(s[:r])@Vh[:r,:]
        Cov=np.diag(Cov.diagonal())+off_diag
        err=linalg.norm(Cov-old_cov)
        if err/(data.shape[0]**2)<limit:
            break
    U,s,Vh=linalg.svd(Cov)
    Gamma=U[:,:r]
    F=np.dot(Gamma.T,data)
    return Gamma,F
        
def eigen_decomp(M):
    eigenValues, eigenVectors = linalg.eig(M)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return np.real(eigenValues),np.real(eigenVectors)
#calculate the trace statistics
def trace_stat(A,B):
    return np.trace(A@B.T@inv(B@B.T)@B@A.T)/np.trace(A@A.T)
#evaluate loading matrix estimate
def compare_loading_plot(Gamma,Gamma_star):
    for i in range(len(Gamma.T)):
        fix,ax=plt.subplots()
        #ax.plot(loading[:,i])
        if r2_score(Gamma[:,i], Gamma_star[:,i])<r2_score(Gamma[:,i], -Gamma_star[:,i]):
            Gamma_star[:,i]=-Gamma_star[:,i]
            #print(i)
        ax.plot(Gamma[:,i])
        ax.plot(Gamma_star[:,i])
        print("r2=",r2_score(Gamma[:,i], Gamma_star[:,i]))     
def compare_factor_plot(F,F_star):
    for i in range(len(F)):
        fix,ax=plt.subplots()
        #ax.plot(loading[:,i])
        if r2_score(F[i], F_star[i])<r2_score(F[i], -F_star[i]):
            F_star[i]=-F_star[i]
            #print(i)
        ax.plot(F[i])
        ax.plot(F_star[i])
        print("r2=",r2_score(F[i], F_star[i]))                                   
def log_multivariate_normal_density(X, means, covars, min_covar=1e-7):
    """Log probability for full covariance matrices. """
    
    solve_triangular = linalg.solve_triangular
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        for i in range(100):
            try:
                cv_chol = linalg.cholesky(cv                                           + i*1e-7 * np.eye(n_dim),lower=True)
                break
            except:
                continue
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = -0.5 * (np.sum(cv_sol ** 2, axis=1)  + cv_log_det)

    return log_prob
def compare_factors(model,ll,true_factor):
    smoothed_state_estimates=model.stacked_factor
    pca_factor=model.pca_factor
    for i in range(model.n_factor):
        pl.figure(figsize=(16, 6))
        if r2_score(true_factor[i][model.lag-1:], smoothed_state_estimates[i])<r2_score(true_factor[i][model.lag-1:], -smoothed_state_estimates[i]):
                true_factor[i][model.lag-1:]=-true_factor[i][model.lag-1:]
        print('r2 for pca estimates=',r2_score(true_factor[i][2:],model.pca_factor[i,2:]), ',  r2 for em estimates=',r2_score(true_factor[i][2:], model.stacked_factor[i]))
        lines_true = pl.plot(true_factor[i][2:], linestyle='-', color='b')
        lines_pca = pl.plot(model.pca_factor[i,2:], linestyle='--', color='r')
        lines_mine = pl.plot(model.stacked_factor[i], linestyle='-.', color='g')
        
        pl.legend(
            (lines_true[0], lines_pca[0], lines_mine[0]),
            ('true', 'pca', 'mine')
        )
        pl.xlabel('time')
        pl.ylabel('state')

    # Draw log likelihood of observations as a function of EM iteration number.
    # Notice how it is increasing (this is guaranteed by the EM algorithm)
    pl.figure()
    pl.plot(ll)
    pl.xlabel('em iteration number')
    pl.ylabel('log likelihood')
    pl.show()    


# In[ ]:




