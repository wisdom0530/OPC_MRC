import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils import utils
from collections import defaultdict
from torch.optim.optimizer import Optimizer, required


class Constraint(torch.optim.Optimizer):
    """
    first_step: gradient of objective 1, and log the grad,
    second_step: gradient of objective 2, and do something based on the logged gradient at step one
    closure: the objective 2 for second step
    """

    def __init__(self, params, base_optimizer, g_star=0.05, alpha=1, beta=1, **kwargs):
        defaults = dict(g_star=g_star, **kwargs)
        super(Constraint, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.g_star = g_star
        self.alpha = alpha
        self.beta = beta
        self.g_constraint = 0.
        self.g_value = torch.tensor([1.]).item()


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                constraint_grad = torch.ones_like(p.grad) * p.grad  # deepcopy, otherwise the c_grad would be a pointer
                self.state[p]["constraint_grad"] = constraint_grad
                
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def dbgd_step(self, zero_grad=False):
        '''
        calculate the projection here
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                phi_x = min( self.alpha * (self.g_value - self.g_constraint), \
                             self.beta * torch.norm(self.state[p]["constraint_grad"]) ** 2 )

                #phi_x =  self.beta * torch.norm(self.state[p]["constraint_grad"]) ** 2 
                    

                adaptive_step_x = F.relu( (phi_x - (p.grad * self.state[p]["constraint_grad"]).sum()) \
                                        / (1e-8 + self.state[p]["constraint_grad"].norm().pow(2)) )

                p.grad.add_(adaptive_step_x * self.state[p]["constraint_grad"])

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def mj_step(self, zero_grad=False):
        '''
        calculate the projection here
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                phi_x = self.alpha * (self.g_value - self.g_constraint)

                if (self.g_value - self.g_constraint) >= 0 \
                    and  (phi_x - (p.grad * self.state[p]["constraint_grad"]).sum()) >= 0:

                    lm = (phi_x - (p.grad * self.state[p]["constraint_grad"]).sum()) \
                         / (1e-8 + self.state[p]["constraint_grad"].norm().pow(2))
                else:
                    
                    lm = 0 

                p.grad.add_(lm * self.state[p]["constraint_grad"])

        if zero_grad: self.zero_grad() 