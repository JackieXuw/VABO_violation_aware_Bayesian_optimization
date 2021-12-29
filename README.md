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
* Run <code>pip install .</code> in this directory. 

## Usage
* The source code is under ./vabo.
* You can run the interactive notebook under ./examples to learn how to use VABO.



