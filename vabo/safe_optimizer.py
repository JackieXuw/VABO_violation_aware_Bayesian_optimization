"""
Implement safe Bayesian optimizer for our test.
"""
import numpy as np
import safeopt
import GPy
from .base_optimizer import *


class SafeBO(BaseBO):

    def __init__(self, opt_problem, safe_BO_config):
        super(SafeBO, self).__init__(opt_problem, safe_BO_config, reverse_meas=True)
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def optimize(self):
        x_next = self.opt.optimize()
        x_next = np.array([x_next])
        return x_next

    def make_step(self):
        x_next, y_obj, constr_vals, vio_cost = self.step_sample_point(reverse_meas=True)
        return y_obj, constr_vals