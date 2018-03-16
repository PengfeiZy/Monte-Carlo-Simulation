import numpy as np


def barber_shop(T=10000, lmd=10, mu=2, gamma=3):
    p_matrix = np.array([[0, 1, 0, 0],
                  [mu/(lmd+mu), 0, lmd/(lmd+mu), 0],
                  [0, (mu+gamma)/(mu+gamma+lmd), 0, lmd/(mu+gamma+lmd)],
                  [0, 0, 1, 0]
                  ])
    numbers, busy, full, dts = [], [], [], []
    t, x = 0, 0

    while t < T:
        if x == 0:
            u = np.random.uniform(0, 1)
            dt = - 1 / lmd * np.log(u)
            t += dt
            x += 1
        elif x == 1:
            p = p_matrix[1]
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            if u1 < p[0]:
                dt = - 1 / lmd * np.log(u2)
                t += dt
                x -= 1
                busy.append(dt)
            else:
                dt = - 1 / mu * np.log(u2)
                t += dt
                x += 1
                busy.append(dt)
        elif x == 2:
            p = p_matrix[2]
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            if u1 < p[1]:
                dt = - 1 / lmd * np.log(u2)
                t += dt
                x -= 1
                busy.append(dt)
            else:
                dt = - 1 / (mu + gamma) * np.log(u2)
                t += dt
                busy.append(dt)
                x += 1
        else:
            u = np.random.uniform(0, 1)
            dt = - 1 / (mu + 2 * gamma) * np.log(u)
            t += dt
            busy.append(dt)
            full.append(dt)
            x -= 1
        numbers.append(x)
        dts.append(dt)
    return np.array(numbers), np.array(dts), busy, full


class OptionPrice(object):
    def __init__(self, k=[80, 100, 120], init_price=100,
                 n_samples=100000, sigma=1, alpha=0.2, t=4):
        self.n_samples = n_samples
        self.sigma = sigma
        self.alpha = alpha
        self.t = t
        self.dt = 0.01
        self.n_times = int(t / self.dt)
        self.stock = np.array([[init_price] * self.n_samples])
        self.k = k

    def stock_price(self):

        mu = (self.alpha - 0.5 * self.sigma**2) * self.dt
        sigma = self.sigma * np.sqrt(self.dt)

        for i in range(self.n_times):
            z = np.random.normal(0, 1, self.n_samples)
            stock_dt = self.stock[i] * np.exp(mu + sigma * z)
            self.stock = np.append(self.stock, [stock_dt], axis=0)

    def euro_option(self):
        final_stock = self.stock[self.n_times]
        options = []
        for i in self.k:
            stock = final_stock[final_stock > i]
            option = np.sum(stock - i) / \
                     self.n_samples * np.exp(-self.alpha * self.t)
            options.append(option)
        return options

    def asian_option(self):
        average_stock = np.mean(self.stock, axis=0)
        options = []
        for i in self.k:
            stock = average_stock[average_stock > i]
            option = np.sum(stock - i) / \
                     self.n_samples * np.exp(-self.alpha * self.t)
            options.append(option)
        return options

    def lookback_option(self):
        stock_min = np.min(self.stock, axis=0)
        final_stock = self.stock[self.n_times]
        payoff = final_stock - stock_min
        payoff = payoff[payoff > 0]
        option = np.sum(payoff) / self.n_samples * \
                  np.exp(-self.alpha * self.t)
        return option

    def call_on_call(self, k1=80, k2=120, t1=2, t2=4):
        final_stock = self.stock[self.n_times] - k2
        final_stock[final_stock < 0] = 0
        middle_option = final_stock * np.exp(-self.alpha * (t2-t1)) - k1
        middle_option[middle_option < 0] = 0
        options = np.mean(middle_option * np.exp(-self.alpha * t1))
        return options


def jump_diffusion(s0=100, alpha=0.2, lmd=1, sigma=1, k=100,
                   n=10000, a=0, b_square=0.5, t=4):
    x = np.array([0] * n)
    mu = alpha - (np.exp(0.5 * b_square) - 1) * lmd
    mu_t = (mu - 0.5 * sigma**2) * t
    sigma_t = sigma * np.sqrt(t)
    M = []
    N = np.random.poisson(lmd*t, n)
    for each in N:
        if each == 0:
            M.append(0)
        else:
            m = np.sum(np.random.normal(a, np.sqrt(b_square), each))
            M.append(m)
    M = np.array(M)
    z = np.random.normal(0, 1, n)
    x = x + mu_t + sigma_t * z + M
    payoffs = s0 * np.exp(x) - k
    payoffs[payoffs < 0] = 0
    option = np.mean(payoffs) * np.exp(-alpha*t)
    return option


def outcomes():
    # Question 1
    numbers, dts, busy, full = barber_shop()
    T = 10000

    # Question 1, (a):
    QT = np.sum(numbers * dts) / T
    print('\nlong-run average number of customers in the shop')
    print(QT)

    # Question 1, (b):
    BT = np.sum(busy) / T
    print('\nlong-run proportion of time the barber is busy')
    print(BT)

    # Question 1, (c):
    FT = np.sum(full) / T
    print('\nlong-run proportion of time the shop is full')
    print(FT)

    # Question 1, (d):
    waiting = numbers - 1
    waiting[waiting < 0] = 0
    LT = np.sum(waiting * dts) / T
    print('\nlong-run average number of customers waiting in line')
    print(LT)

    # Question 2
    option = OptionPrice()
    option.stock_price()

    # Question 2, (b):
    euro_calls = option.euro_option()
    print('\nEuropean call with strike price = 80, 100, 120:')
    print(euro_calls)
    # Question 2, (c):
    asian_calls = option.asian_option()
    print('\nAsian call with strike price = 80, 100, 120:')
    print(asian_calls)

    # Question 2, (d):
    lookback_calls = option.lookback_option()
    print('\nLoocback option:')
    print(lookback_calls)

    # Question 3:
    call_on_call = option.call_on_call()
    print('\nCall on call option price:')
    print(call_on_call)

    # Question 4:
    options = jump_diffusion()
    print('\nJump diffusion call option price:')
    print(options)
