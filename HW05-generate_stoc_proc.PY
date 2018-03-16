import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# Question 1: Simulate the Poisson path

def poisson_process(lmd=5, T=2):
    t, number = 0, 0
    numbers, time = [0], [0]
    while t <= T:
        number += np.random.randint(20, 40)
        u = np.random.uniform(0, 1)
        t = t - 1 / lmd * np.log(u)
        time.append(t)
        numbers.append(number)
    plt.plot(time, numbers, drawstyle='steps-post')
    plt.xlim(0, 2)
    plt.show()


# Question 2: NHPP with 2 different algorithms

def poisson_NHPP(T=20, naive=True, n=100):
    numbers, paths = [], {}
    lmd = lambda t: 1 + 0.6 * np.sin(t)
    lmd_max = 1.6

    if naive:
        for i in range(n):
            num, t, number, time = 0, 0, [], []
            while t <= T:
                time.append(t)
                number.append(num)
                u = np.random.uniform(0, 1)
                t = t - 1 / lmd(t) * np.log(u)
                num += 1
            numbers.append(num)
            paths[i] = np.array([time, number])

    else:
        for i in range(n):
            num, t, number, time = 0, 0, [0], [0]
            while t <= T:
                u1 = np.random.uniform(0, 1)
                u2 = np.random.uniform(0, 1)
                p = lmd(t) / lmd_max
                t = t - 1 / lmd_max * np.log(u1)
                if u2 <= p and t <= 20:
                    time.append(t)
                    num += 1
                    number.append(num)
            numbers.append(num)
            paths[i] = np.array([time, number])

    return numbers, paths


def confidence_interval(naive1=True):
    numbers, paths = poisson_NHPP(naive=naive1)
    time = [i for i in np.arange(0, 20, 0.01)]
    means, stds = [], []

    for t in time:
        tol = []
        for each in paths:
            number = paths[each][1][paths[each][0] <= t][-1]
            tol.append(number)
        means.append(np.mean(tol))
        stds.append(np.std(tol) / 10)

    up_95 = []
    low_95 = []
    for i in range(len(means)):
        low_95.append(means[i] - 1.96 * stds[i])
        up_95.append(means[i] + 1.96 * stds[i])

    total_mean = np.mean(numbers)
    total_variance = np.var(numbers)

    return total_mean, total_variance, up_95, low_95


def plot_paths(name, naive2=True):
    total_mean, total_var, up_95, low_95 = confidence_interval(naive1=naive2)

    lmd = lambda x: x + 0.6 - 0.6 * np.cos(x)
    time = [i for i in np.arange(0, 20, 0.01)]
    actual = [lmd(i) for i in time]

    plt.plot(time, low_95, '-')
    plt.plot(time, up_95, '--')
    plt.plot(time, actual, ':')
    plt.legend(['low 95 CI', 'up 95 CI', 'actual value'])
    plt.title('{0} simulation graph'.format(name))
    plt.xlabel('time')
    plt.ylabel('total numbers')
    plt.show()



# Question 3
def discretetimemc(n=1000, x=0):
    s = {0: 0, 1: 0, 2:0}
    xs = []
    p_matrix = {0: [1/2, 1/3, 1/6],
         1: [0, 1/3, 2/3],
         2: [1/2, 0, 1/2]}

    while n:
        u = np.random.uniform(0, 1)
        p = p_matrix[x]
        if u < p[0]:
            x = 0
        elif u < p[1] + p[0]:
            x = 1
        else:
            x = 2

        s[x] += 1
        xs.append(x)
        n -= 1
    p0 = s[0] / 1000
    p1 = s[1] / 1000
    p2 = s[2] / 1000
    meanx = np.mean(xs)

    print('\nThe probability of x=0, 1, 2 are seperately: {0}, {1}, {2}\n'
          .format(p0, p1, p2))
    print('The mean of Xs is: {}'.format(meanx))

# Question 1
poisson_process()

# Question 2
plot_paths('naive')
plot_paths('NHPP', naive2=False)

# Question 3
discretetimemc()
discretetimemc()