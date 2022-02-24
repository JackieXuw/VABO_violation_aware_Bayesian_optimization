"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
from .base_optimizer import BaseBO
from scipy.stats import norm


class ViolationAwareBO(BaseBO):

    def __init__(self, opt_problem, violation_aware_BO_config):
        # optimization problem and measurement noise
        super().__init__(opt_problem, violation_aware_BO_config)

        # Pr(cost <= beta * budget) >= 1 - \epsilon
        if 'beta_func' in violation_aware_BO_config.keys():
            self.beta_func = violation_aware_BO_config['beta_func']
        else:
            self.beta_func = lambda t: 1

        self.num_eps = 1e-10   # epsilon for numerical value
        self.total_vio_budgets = violation_aware_BO_config['total_vio_budgets']
        self.prob_eps = violation_aware_BO_config['prob_eps']
        self.beta_0 = violation_aware_BO_config['beta_0']
        self.total_eval_num = violation_aware_BO_config['total_eval_num']

        self.curr_budgets = self.total_vio_budgets
        self.curr_eval_budget = self.total_eval_num
        self.single_step_budget = violation_aware_BO_config[
               'single_max_budget']

        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)
        self.S = None

    def get_acquisition(self, prob_eps=None):
        if prob_eps is None:
            prob_eps = self.prob_eps
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)
        obj_mean = obj_mean + self.gp_obj_mean
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)
            mean = mean + self.gp_constr_mean_list[i]
            constrain_mean_list.append(np.squeeze(mean))
            constrain_var_list.append(np.squeeze(var))

        constrain_mean_arr = np.array(constrain_mean_list).T
        constrain_var_arr = np.array(constrain_var_list).T

        # calculate Pr(g_i(x)<=0)
        prob_negative = norm.cdf(0, constrain_mean_arr, constrain_var_arr)
        # calculate feasibility prob
        prob_feasible = np.prod(prob_negative, axis=1)

        # calculate EI and EIc
        f_min = self.best_obj
        z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
        EI = (f_min - obj_mean) * norm.cdf(z) + np.sqrt(obj_var) * norm.pdf(z)
        EIc = prob_feasible * EI

        # calculate Pr(c_i([g_i(x)]^+)<=B_{i,t} * beta_{i, t})
        curr_beta = self.get_beta()
        curr_cost_allocated = self.curr_budgets * curr_beta
        allowed_vio = self.opt_problem.get_vio_from_cost(curr_cost_allocated)
        prob_not_use_up_budget = norm.cdf(allowed_vio, constrain_mean_arr,
                                          constrain_var_arr)
        prob_all_not_use_up_budget = np.prod(prob_not_use_up_budget, axis=1)

        EIc_indicated = EIc * (prob_all_not_use_up_budget >=
                               1 - prob_eps)

        self.S = self.parameter_set[(prob_all_not_use_up_budget >=
                                     1 - prob_eps)]
        return EIc_indicated

    def get_beta(self):
        return min(max(0, self.beta_func(self.curr_eval_budget)), 1.0)

    def optimize(self):
        prob_eps = self.prob_eps
        eps_multi = 1.1
        is_any_acq_postive = False
        while not is_any_acq_postive:
            acq = self.get_acquisition(prob_eps=prob_eps)
            if np.any(acq > 0):
                is_any_acq_postive = True
            else:
                print('Can not find not use up budget point, increase risk \
                      level.')
                prob_eps = prob_eps * eps_multi
        next_point_id = np.argmax(acq)
        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self, update_gp=False, gp_package='gpy'):
        x_next, y_obj, constr_vals, vio_cost = self.step_sample_point()
        vio_cost = np.squeeze(vio_cost)
        self.cumu_vio_cost = self.cumu_vio_cost + vio_cost
        self.curr_budgets = np.minimum(
            np.maximum(self.total_vio_budgets - self.cumu_vio_cost, 0),
            self.single_step_budget
        )
        self.curr_eval_budget -= 1
        return y_obj, constr_vals
