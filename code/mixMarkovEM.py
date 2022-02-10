import scipy.io
import pandas as pd
import math
import numpy as np
from numpy import matlib as mb
from scipy.sparse import csr_matrix
from scipy.stats import zscore
from collections import Counter

def unpack_sample_data():
    mat = scipy.io.loadmat('sample_data.mat')
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    for k, v in mat.items():
        if k == "x":
            data = np.array([np.ravel(i[0]) for i in v])
            # print([np.ravel(i[0]) for i in v])
            # data = pd.DataFrame({k: pd.Series([np.ravel(i[0]) for i in v])})
            # data.to_csv("sample_data.csv",index=None)
            # print(k)
            # print(v.shape)
        # else:
            # p1, z, A
            # print(k)
            # print(v.shape)
            # print(v)
    return data

# def read_sample_data():
#     data = pd.read_csv('sample_data.csv')
#     print(data.head(5))
#     return data

def logsafe(x, cutpoint=-500):
    """
    Takes the logarithm of x, avoiding the values to go to -Inf for numerical stability.
    """
    log_x = np.log(x)
    log_x = np.nan_to_num(log_x)
    return log_x

def normalize(input_array):
    """
    Column-wise Normalization: normalize elements of each col by column_sum
    """
    col_sum = np.sum(input_array, axis=0)
    input_array = input_array + (input_array==0)
    dm = np.ones((1,int(len(input_array.shape))))
    dm[0][0] = int(input_array.shape[0])
    if len(input_array.shape) == 3:
        input_array = input_array/np.array([mb.repmat(e,int(dm[0][0]),int(dm[0][1])) for e in col_sum])
    else:
        input_array = input_array/mb.repmat(col_sum,int(dm[0][0]),int(dm[0][1]))
    return input_array


def normalize_exp(input_array):
    """
    Numerically stable exp + normalize. Applies to the 1st dimension of tensor X
    """
    col_max = np.amax(input_array,axis=0)
    dm = np.ones((1,int(len(input_array.shape))))
    dm[0][0] = int(input_array.shape[0])
    input_array = input_array - np.array(mb.repmat(col_max,int(dm[0][0]),int(dm[0][1])))
    input_array = normalize(np.exp(input_array))
    return input_array

# def log_sum_exp(input_array):
# """
# log(sum(exp(X))) operation with avoiding underflow
# alternative: sum(sum(llhood .* z)) - sum(sum(z .* logsafe(z)));
# """
#     dm = ones(1, length(size(X)));
#     dm(1) = size(X,1);
#     mx = max(X);
#     mx2 = repmat(mx, dm);
#     L = mx + log(sum(exp(X-mx2)));


def mixMarkovEM(x=None,D=None,K=None,*args,**kwargs):
  """
  :input param x : cell array of sequences
  :input param D : dimension of the state space.
  :input param K : number of mixture components. (>1)
  :input param maxIter: Maximum EM iteration count (default 100)
  :input param verbose': plots log likelihood (default 'false')
  :input param threshold': if the decrease in log likelihood is less than the threshold, stop.
  :output alpha: prior distribution of the mixture components p(z)
  :output p1: cond. dist.of initial states p(x_n(1)|z_n)
  :ouput A: cond. dist. of state transitions p(x_n(t)|x_n(t-1),z_n)
  :output z: posterior distribution of mixture componenets p(z|x,pi,A)
  :output LL: the bound value during the computations

  Sample Usage:
  [alpha,pi,A,z,LL] = bk_mixMarkovEM(y,D,K,'maxIter', 1000, 'verbose',true,'threshold', 1e-5)
  1. Runs the EM algorithm for the sequences y, whose input space dimension is D, and the number of mixture components are K.
  2. The algorithm is run 1000 times unless the decrease in log likelihood is less than 1e-5.
  3. The loglikelihood is also plotted after every iteration.
  """
  maxIter= 1000
  verbose= True
  threshold = 1e-5
  N = x.shape[0] # number of sequences in observations
  # print("N: {}".format(N))
  S = np.array([np.zeros((D,D)) for _ in range(N)])# for transition matrix of each sequences
  # print("S: {}".format(S[0]))
  x1 = np.zeros((D,N)) # for states of each sequences
  # print("x1: {}".format(x1))
  for n in range(N):
      x1[x[n][0]-1,n]=1 # create matrix for the first state of each sequence
      cur_pre_states = list(zip(x[n][1:],x[n][:-1]))
      freq = list(dict(Counter(e for e in cur_pre_states)).values())
      row = [e[0]-1 for e in list(dict(Counter(e for e in cur_pre_states)).keys())]
      col = [e[1]-1 for e in list(dict(Counter(e for e in cur_pre_states)).keys())]
      S[n] = csr_matrix((freq, (row, col)), shape=(D, D)).toarray()
  # print(pd.DataFrame(x1))
  S = np.transpose([i.flatten('F') for i in S])
  # print(S)
  alpha = normalize(np.reshape(np.random.uniform(0,1,K),(K,1),order="F"))
  # print(alpha)
  p1=normalize(np.reshape(np.random.uniform(0,1,D*K),(D,K),order="F"))
  # print(p1)
  A=normalize(np.reshape(np.random.uniform(0,1,D*D*K),(D,D,K),order="F"))
  # print(A)
  z = np.zeros((K,N)) # the posterior of latent variables
  LL = np.zeros((1,maxIter)) # loglikelihood

  for iter in range(maxIter):
      log_p1 = logsafe(p1)
      log_A = np.transpose([i.flatten('F') for i in logsafe(A)])
      log_alpha = logsafe(alpha)

      # E-step:
      llhood = np.dot(np.transpose(log_p1),x1) + np.dot(np.transpose(log_A),S) + mb.repmat(log_alpha,1,N)
      z = normalize_exp(llhood)
      # Likelihood:
      LL[0][iter] = sum(sum(np.multiply(llhood,z))) - sum(sum(np.multiply(z,logsafe(z))))
      # print(LL)
      # M-step:
      p1 = normalize(np.dot(x1,np.transpose(z)))
      A = normalize(np.reshape(np.dot(S,np.transpose(z)),(D,D,K), order="F"))
      alpha = normalize(np.sum(z,axis=1).reshape(K,1))

      if (threshold > 0) and (iter >1) and (LL(iter) - LL(iter-1) < threshold):
          LL = LL[0][1:iter]
      break

  return alpha,p1,A,z,LL,S




if __name__ == "__main__":
    data = unpack_sample_data()
    print(data.shape)
    print(data[0])
    alpha,p1,A,z,LL,S = mixMarkovEM(data,4,4)
    print(alpha)
