#!/usr/bin/env python
# coding: utf-8

# In[12]:


from DFM_class import *
from functions import *
from RPCA_ALM_class import*
from SPCP import *
import seaborn as sns
import numpy as np
np.random.seed(0)


# In[10]:


y,Gamma,factor,noise=generate_test_data(1000,1000,1)


# In[11]:


common=Gamma@factor
print((common@common.T/1000).diagonal().mean())
print((noise@noise.T/1000).diagonal().mean())


# In[ ]:


n_lst=[20,40]
T=1000
sigma_lst=[1,2]
loading_tr={}
factor_tr={}
R2={}
for i,N in enumerate(n_lst):
    loading_tr[N]={}
    factor_tr[N]={}
    R2[N]={}
    for j,sigma in enumerate(sigma_lst):
        #T=N*10
        ltr=[]
        ftr=[]
        r2=[]
        for k in range(2):

            y,Gamma,factor,noise=generate_hetero_test_data(N,T,sigma)
            model=DFM(y,0,5)
            model.pca()
            model.to_state_space_rep()
            model.em(max_iter=1000)
            #ltr.append(trace_stat(Gamma.T,model.pca_loading.T))
            #ftr.append(trace_stat(factor,model.pca_factor))
            r2.append(max(r2_score((Gamma@factor).T,model.pca_common.T),r2_score((Gamma@factor[:,model.lag-1:]).T,model.stacked_common.T)))
            #r2.append(r2_score((Gamma@factor[:,model.lag-1:]).T,(model.stacked_loading@model.stacked_factor).T))
        #loading_tr[N][sigma]=np.array(ltr).mean()
        print(N,sigma,model.n_factor, np.array(r2).mean())
        R2[N][sigma]=np.array(r2).mean()


# In[6]:


df=pd.DataFrame.from_dict(R2)
ax = sns.heatmap(df, vmin=0, vmax=1)
df.to_csv(f'tests/hetero_R2_T={T}.csv')
df


# In[4]:


df=pd.DataFrame.from_dict(R2)
ax = sns.heatmap(df, vmin=0, vmax=1)
df.to_csv(f'tests/R2_T={T}.csv')
df


# In[14]:


n_lst=[20,40,80,160,320,640,1280,2560]
T=1000
sigma_lst=[1,2,3,4,5,6,7,8,9]
loading_tr={}
factor_tr={}
R2={}
for i,N in enumerate(n_lst):
    loading_tr[N]={}
    factor_tr[N]={}
    R2[N]={}
    for j,sigma in enumerate(sigma_lst):
        #T=N*10
        ltr=[]
        ftr=[]
        r2=[]
        for k in range(10):

            y,Gamma,factor,noise=generate_test_data(N,T,sigma)
            yao=Yao(y,5)
            yao.fit()

            model=DFM(yao.residual,0,5)
            model.pca()
            #model.to_state_space_rep()
            #model.em(max_iter=1000)
            #ltr.append(trace_stat(Gamma.T,model.pca_loading.T))
            #ftr.append(trace_stat(factor,model.pca_factor))
            r2.append(r2_score((Gamma@factor).T,(y-yao.residual+model.pca_common).T))
            #r2.append(r2_score((Gamma@factor[:,model.lag-1:]).T,(model.stacked_loading@model.stacked_factor).T))
        #loading_tr[N][sigma]=np.array(ltr).mean()
        print(N,sigma,model.n_factor, np.array(r2).mean())
        R2[N][sigma]=np.array(r2).mean()


# In[12]:


from sklearn.metrics import mean_absolute_error
#test for different N,sigma
R2=[]
#specify number of factors
n_factor=5
for n in [10,20,40,80,160,320,640]:
    mae=[]
    for k in range(1):
        #generate test data
        X,Gamma,F,noise=generate_vector_test_data(n,1000,10)
        #initialize mode
        model=DFM(X,n_factor,5)
        #initialize parameter using pca
        model.pca()
        #map model to stacked form
        #model.to_state_space_rep()
        #EM estimation, loglikylihood is returned and saved to ll.
        #ll=model.em(max_iter=1000)
        #y_true=F[:,model.lag-1:]
        #y_pred=model.stacked_factor[:5]
        #for i in range(5):
            #if r2_score(y_true[i], y_pred[i])<r2_score(y_true[i], -y_pred[i]):
               # y_pred[i]=-y_pred[i]
        #mae.append(mean_absolute_error(y_true.T, y_pred.T))
        mae.append(trace_stat(Gamma.T,model.pca_loading.T))
    R2.append(np.array(mae).mean())


# In[13]:


plt.plot([10,20,40,80,160,320,640],R2)
plt.plot([10,20,40,80,160,320,640],1/np.sqrt([10,20,40,80,160,320,640]))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('number of stocks N')
plt.ylabel('R2')
plt.legend(['R2', '1/sqrt(N)'])


# In[38]:


from sklearn.metrics import mean_absolute_error
#test for different N,sigma
R2=[]
#specify number of factors
n_factor=5
for T in [100,200,400,800,1600,3200,6400]:
    r2=[]
    for k in range(1):
        #generate test data
        X,Gamma,F,noise=generate_vector_test_data(1000,T,5)
        #initialize mode
        model=DFM(X,n_factor,5)
        #initialize parameter using pca
        model.pca()
        #map model to stacked form
        #model.to_state_space_rep()
        #EM estimation, loglikylihood is returned and saved to ll.
        #ll=model.em(max_iter=1000)
        #y_true=F[:,model.lag-1:]
        #y_pred=model.stacked_factor[:5]
        #for i in range(5):
            #if r2_score(y_true[i], y_pred[i])<r2_score(y_true[i], -y_pred[i]):
               # y_pred[i]=-y_pred[i]
        #mae.append(mean_absolute_error(y_true.T, y_pred.T))
        r2.append(trace_stat(Gamma.T,model.pca_loading.T))
    R2.append(np.array(r2).mean())


# In[41]:


plt.plot([100,200,400,800,1600,3200,6400],R2)
plt.plot([100,200,400,800,1600,3200,6400],1/np.sqrt([100,200,400,800,1600,3200,6400]))
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('number of time steps T')
plt.ylabel('mae')
plt.legend(['mae', '1/sqrt(T)'])


# In[ ]:
