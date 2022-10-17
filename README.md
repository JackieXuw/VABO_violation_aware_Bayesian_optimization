# VABO: Violation-Aware Bayesian Optimization

This repo implements a version of the violation-aware Bayesian optimization.
Violation-aware Bayesian optimization is a derivative-free and sample-efficient method that
automatically tunes parameters of a system to optimize a performance/cost metric while keeping the cost of 
constraint violation under prescribed budgets.

If you use the repo for your research, we appreciate it that you can cite the paper:

@article{xu2021vabo,
  title={VABO: Violation-Aware Bayesian Optimization for Closed-Loop Control Performance Optimization with Unmodeled Constraints},
  author={Xu, Wenjie and Jones, Colin N and Svetozarevic, Bratislav and Laughman, Christopher R and Chakrabarty, Ankush},
  journal={arXiv preprint arXiv:2110.07479},
  year={2021}
}

## Install
* <code>git clone https://github.com/JackieXuw/VABO_violation_aware_Bayesian_optimization.git </code>
* Run <code>pip install .</code> in this directory. 

## Usage
* The source code is under ./vabo.
* You can run the interactive notebook under ./examples to learn how to use VABO.
* To construct an optimization problem. You need to provide a config dictionary
    to initialize an object of the class `vabo.optimization_problem.OptimizationProblem`. 
  * For example.
    ```python
    config = {
    'problem_name': 'my_problem',   # the name of your problem
    'var_dim': 5,   # the dimension of input variables
    'discretize_num_list': [5, 5, 5, 5, 5],  # the discretizing number you want along each direction
    'num_constrs': 1,   # the number of constraints
    'bounds': [(100, 300), (300, 500), (300, 500), (300, 500), (100, 200)],  # the bounds for the input variables
    'obj': <function>,  # the optimization objective functions
    'constrs_list': <function>,  # the optimization constraints functions
    'vio_cost_funcs_list': <function>,  # the function mapping violations to the cost
    'vio_cost_funcs_inv_list': <function>,  # the inverse mapping from cost to violation
    'init_safe_points': array([[300, 500, 500, 500, 200]]),  # an array of initial safe points
    'train_X': array([[300, 500, 500, 500, 200]]),  # an array of training points to learn the kernel
    'kernel': [<GPy.kern.src.rbf.RBF>,
               <GPy.kern.src.rbf.RBF>],  # the array of kernel functions to model the unknown functions
    'single_max_budget': 1600  # the maximum allowed violation cost in one single step
    }
    ```

