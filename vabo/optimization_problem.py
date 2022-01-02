import numpy as np
import safeopt
"""
Define and implement the class of optimization problem.
"""


class OptimizationProblem:

    def __init__(self, config):
        self.config = config
        self.evaluated_points_list = []
        self.evaluated_objs_list = []
        self.evaluated_constrs_list = []
        self.problem_name = config['problem_name']

        # if obj and constr are evaluated simultaneously using a simulator
        self.eval_simu = config['eval_simu']

        if self.eval_simu:
            self.simulator = None

        self.var_dim = config['var_dim']
        self.num_constrs = config['num_constrs']
        self.obj = config['obj']
        self.constrs_list = config['constrs_list']
        self.constrs_vio_cost_funcs_list = config['vio_cost_funcs_list']
        self.vio_cost_funcs_inv_list = config['vio_cost_funcs_inv_list']
        self.bounds = config['bounds']
        self.discretize_num_list = config['discretize_num_list']
        self.init_safe_points = config['init_safe_points']
        self.train_X = config['train_X']
        self.train_obj, self.train_constr = self.sample_point(self.train_X)
        self.candidates = safeopt.\
            linearly_spaced_combinations(self.bounds, self.discretize_num_list)

    def get_minimum(self):
        obj_val, constr = self.sample_point(self.candidates)
        obj_val = obj_val.squeeze()
        feasible = np.array([True] * len(obj_val))
        for i in range(self.num_constrs):
            feasible = feasible & (constr[:, i] <= 0)

        minimum = np.min(obj_val[feasible])
        feasible_candidates = self.candidates[feasible, :]
        minimizer = feasible_candidates[np.argmin(obj_val[feasible]), :]
        return minimum, minimizer

    def sample_point(self, x):
        if self.eval_simu:
            obj_val, constraint_val_arr, simulator = self.obj(x,
                                                              self.simulator)
        else:
            obj_val = self.obj(x)
            obj_val = np.expand_dims(obj_val, axis=1)
            constraint_val_list = []
            for g in self.constrs_list:
                constraint_val_list.append(g(x))
            constraint_val_arr = np.array(constraint_val_list).T

        self.evaluated_points_list.append(x)
        self.evaluated_objs_list.append(obj_val)
        self.evaluated_constrs_list.append(constraint_val_arr)
        return obj_val, constraint_val_arr

    def get_total_violation_cost(self, constraint_val_arr):
        constrs_vio_cost_list = []
        for i in range(self.num_constrs):
            cost_func = self.constrs_vio_cost_funcs_list[i]
            vio_cost = cost_func(np.maximum(constraint_val_arr[:, i], 0))
            constrs_vio_cost_list.append(vio_cost)
        return np.array(constrs_vio_cost_list).T

    def get_vio_from_cost(self, cost_budget):
        assert np.all(cost_budget >= 0)
        allowed_vio = np.zeros((1, self.num_constrs))
        for i in range(self.num_constrs):
            c_inv = self.vio_cost_funcs_inv_list[i](cost_budget[i])
            allowed_vio[0, i] = c_inv
        return allowed_vio
