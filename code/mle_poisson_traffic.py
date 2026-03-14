import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Observed traffic data (vehicles per minute)
traffic_data = np.array([12,15,10,8,14,11,13,16,9,
                         12,11,14,10,15])

# Negative log likelihood for Poisson
def negative_log_likelihood(lam, data):
    """
    Computes the negative log-likelihood for a Poisson model.
    log(k!) is ignored because it does not depend on lambda.
    """

    log_likelihood = np.sum(data * np.log(lam) - lam)

    return -log_likelihood


# Initial guess for lambda
initial_guess = 1.0

# Numerical optimization
result = opt.minimize(
    negative_log_likelihood,
    initial_guess,
    args=(traffic_data,),
    bounds=[(0.001, None)]
)

mle_lambda = result.x[0]

print("Numerical MLE lambda:", mle_lambda)
print("Analytical lambda:", np.mean(traffic_data))


# Visualization
lam = np.mean(traffic_data)

x = np.arange(0,25)

pmf = poisson.pmf(x, lam)

plt.hist(traffic_data,
         bins=8,
         density=True,
         label="Observed Data")

plt.plot(x, pmf,
         'o-',
         label="Poisson Model")

plt.xlabel("Cars per minute")
plt.ylabel("Probability")
plt.title("Traffic Model using Poisson Distribution")
plt.legend()

plt.show()
