import numpy as np
from math import exp, sqrt
import seaborn as sns
from scipy import stats
from matplotlib import style
import matplotlib.pyplot as plt
from numpy.random import rand
style.use('ggplot')

# Problem 1.
def pair_dice(n=10000):
    total_num = []
    while n:
        judge = [i for i in range(13) if i > 1]
        roll_num = 0
        while judge:
            total = np.random.randint(1, 7) + np.random.randint(1, 7)
            if total in judge:
                judge.remove(total)
            roll_num += 1
        total_num.append(roll_num)
        n -= 1
    average = np.mean(total_num)
    print('\nProblem 1 (a), average roll numbers are: {}\n'.format(average))


# Problem 3, (a).
def expectation_N(n):
    a, N =n, []
    while n:
        u = 0
        count = 0
        while u <= 1:
            u += np.random.uniform(0, 1)
            count += 1
        N.append(count)
        n -= 1
    average = np.mean(N)
    print('\nProblem 3 (a), E(N) when n = {0} is: {1}\n'.format(a, average))


# Problem 3, (b).
def expecctationM(n, lmd):
    threshold  = exp(-lmd)
    a, M = n, []
    while n:
        u = 1
        count = 0
        while u >= threshold:
            u *= np.random.uniform(0, 1)
            count += 1
        M.append(count-1)
        n -= 1
    average = np.mean(M)
    print('\nProblem 3 (b), E(M) when M = {0} and lambda = {1} is: {2}\n'.format(a, lmd, average))


# Question 4, (c).
def discrete_inverse(q, n=10000):
    N = []
    while n:
        u = np.random.uniform(0, 1)
        count = 0
        for each in q:
            count += 1
            if u <= each:
                N.append(count)
                break
        n -= 1
    average = np.mean(N)
    print('\nProblem 4 (c) and (e):')
    print('E(N) for q = {0} is: {1}\n'.format(q, average))


# Question 5, (c).
def majorizing(x):
    if 0 <= x < 0.36:
        return 3.186 * x
    elif 0.36 <= x < 0.84:
        return 2.074
    elif 0.84 <= x <= 1:
        return -4.461 * x + 4.461
    else:
        return 0


def inverse(x):
    if 0 <= x < 0.2065:
        return sqrt(x / 1.593)
    elif 0.2065 <= x < 1.2020:
        return (x + 0.5401) / 2.074
    elif 1.2020 <= x <= 1.275:
        return -sqrt(((x + 1.569) / (-2.844) +1)) + 1
    else:
        return 0


def accept_reject(n=10000):
    f = lambda x: 60 * x**3 * (1-x)**2
    accept = []
    while n:
        u1 = np.random.uniform(0, 1)
        u2 = 1.275 * np.random.uniform(0, 1)
        y = inverse(u2)
        fy, gy = f(y), majorizing(y)
        if u1 <= fy / gy:
            accept.append(y)
        n -= 1
    sns.distplot(accept, bins=50, kde=False, fit=stats.beta)
    plt.title('ARM-Problem 5')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot
    plt.show()


pair_dice()

for n in [100, 1000, 10000]:
    expectation_N(n)

for lmd in [1, 2]:
    for n in [100, 1000, 10000]:
        expecctationM(n, lmd)

q =[ [0.05, 0.1, 0.2, 0.3, 0.9, 1], [0.6, 0.7, 0.8, 0.9, 0.95, 1]]
for each in q:
    discrete_inverse(each)

accept_reject()


