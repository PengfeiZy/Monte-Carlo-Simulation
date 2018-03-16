import numpy as np
from scipy.stats import norm, gamma, lognorm
import matplotlib.pyplot as plt


class GaussianCopulas(object):

    def __init__(self, rho=0.2, num=100000, mean=[0, 0]):
        self.rho = rho
        self.cov = np.matrix([[1, rho], [rho, 1]])
        self.n = num
        self.mean = np.array(mean)
        self.z = np.random.multivariate_normal(self.mean, self.cov, self.n)
        self.u = norm.cdf(self.z)

    def expweibull(self, lmd=2, alpha=2, beta=3):
        x = (-1/lmd) * np.log(self.u[:, 0])
        y = beta * (-np.log(self.u[:, 1])) ** (1 / alpha)
        corr = np.corrcoef(x, y)
        print('Correlation of Exponential-Weibull when rho={0} is: {1}\n'.format(self.rho, corr[0][1]))

        scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
        x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2, sharex=scatter_axes)
        y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=scatter_axes)

        scatter_axes.plot(x, y, '.')
        x_hist_axes.hist(x, bins=50)
        y_hist_axes.hist(y, bins=50, orientation='horizontal')
        x_hist_axes.set_title('Exponential-Weibull, correlation={}'.format(self.rho))
        plt.show()

    def gammalognormal(self, alpha=6.7, lmd=3, mu=0.1, sigma=0.5):
        x = gamma.ppf(self.u[:, 0], alpha, lmd)
        y = lognorm.ppf(self.u[:, 1], mu, sigma)
        corr = np.corrcoef(x, y)
        print('Correlation of Gamma-Lognorma when rho={0} is: {1}\n'.format(self.rho, corr[0][1]))

        scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
        x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2, sharex=scatter_axes)
        y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=scatter_axes)

        scatter_axes.plot(x, y, '.')
        x_hist_axes.hist(x, bins=50)
        y_hist_axes.hist(y, bins=50, orientation='horizontal')
        x_hist_axes.set_title('Gamma-Lognormal, correlation={}'.format(self.rho))
        plt.show()


rho_first = GaussianCopulas()
rho_first.expweibull()
rho_first.gammalognormal()


rho_second = GaussianCopulas(0.8)
rho_second.expweibull()
rho_second.gammalognormal()


rho_third = GaussianCopulas(-0.8)
rho_third.expweibull()
rho_third.gammalognormal()