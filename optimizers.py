import numpy as np
import scipy.stats as stats


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):
    def __init__(self, sol_dim, max_iters, popsize, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        if num_elites > popsize:
            raise ValueError(
                "Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, evaluator):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert len(init_mean.shape) == 1, "init_mean shape {}".format(init_mean.shape)
        assert len(init_var.shape) == 1, "init_var shape {}".format(init_var.shape)
        mean, var = init_mean, init_var
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        for _ in range(self.max_iters):
            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            samples = samples.astype(np.float32)
            
            costs = evaluator(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]

            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)

            if np.max(var) < self.epsilon:
                break

        return mean
