
# coding: utf-8

# In[75]:


import numpy as np

from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name() # get the name of the node
# In[2]:


lr = 0.05
epoch = 100

def de_exchange(args):
    for node_index in range(size):
        send_req = comm.isend(args,dest=(rank+1)%size)
        res = comm.recv(source=(rank-1)%size)
        send_req.wait()
        args = (args+res)/2

# In[71]:


class dsgd(object):
    def __init__(self,func,der_func,start_point,lr=0.05,epoch=100):
        self.func = func
        self.der_func = der_func
        self.start_point = np.asarray(start_point)
        self.current_solution = np.asarray(start_point)
        self.lr = lr
        self.epoch = epoch
    def forward(self,args):
        return self.func(args)
    def cal_der(self,args):
        return self.der_func(args)
    def gd(self,args):
        self.current_solution = self.current_solution - lr*self.cal_der(args)
        return self.forward(args)
    def process(self):
        for time in range(self.epoch):
            print("processor %s loss : %f ---%d"%\
            (node_name,self.gd(np.asarray(self.current_solution)),time))
            self.exchange()
    def exchange(self):
        send_req = comm.isend(self.current_solution,dest=(rank+1)%size)
        res = comm.recv(source=(rank-1)%size)
        send_req.wait()
        self.current_solution = (self.current_solution+res)/2


# In[72]:


def func(args):
    return np.asarray(args[0]**2+args[1]**3)
def der_func(args):
    return np.asarray(2*args[0],3*args[1]**2)
test = dsgd(func,der_func,(10,10))


# In[73]:


test.process()


# In[38]:


def f(*args):
    print(args)


# In[35]:


f((1,1))


# In[55]:


(np.asarray([1,2]) - np.asarray([2,3]))**2

