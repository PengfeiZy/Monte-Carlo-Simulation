import numpy.random as random
from statistics import mean
from math import exp, log, sqrt

# problem 1.
def lst_to_num(lst):
    num = 0
    for i, digit in enumerate(lst):
        num+=(digit*10**(3-i))
    return num
        

def midsquare_generator(z0=7182, num=20):
    z = [z0]
    i = 0
    while i < num:
        zi = z0**2
        lst = [int(j) for j in str(zi)]
        l = len(lst)
        if l !=8:
            short = 8 -l
            while short > 0:
                lst = [0] + lst
                short-=1
        num_list = lst[2:6]
        z0 = lst_to_num(num_list)
        z.append(z0)
        i+=1
    if z0 == 7182:
        print('Problem 1, (a):', z)
    else:
        print('Problem 1, (b):', z)
    return z

# problem 2.
def lcg_generagor(a=5, c=3, m=16, z0=7, num=20):
    z = [z0]
    i = 0
    while i < num:
        z0 = (a*z0 + c) % m
        z.append(z0)
        i+=1
    print('LCG generator:', z)
    return z

# problem 3.
def pseudo_generator(a=3, b=5, x1=23, x2=66, num=14):
    x = [x1, x2]
    i = 0
    while i < num:
        xi = (a * x[-1] + b * x[-2]) % 100
        x.append(xi)
        i+=1
    print('Pseudo generator:', x)
    return x

def get_rand(num=10000):
    rand_num1 = random.uniform(0,1,num)
    rand_num2 = random.uniform(0,1,num)
    return [rand_num1, rand_num2]
    
x1, x2 = get_rand()

# problem 4.
def integrate_simulate(func, dimension=1, num=10000, rand_num1 = x1,
                       rand_num2 = x2, related=False, relation=None):
    if dimension == 1:
        integration = mean(list(map(func, rand_num1)))
    elif not related:
        integration = mean(list(map(func, rand_num1, rand_num2)))
    else:
        integration = mean(list(map(func, rand_num1,
                                    list(map(relation, rand_num2)))))
    return integration

# (a)
a = integrate_simulate(lambda x: exp(exp(x)))
# (b)
b = integrate_simulate(lambda x: 4*exp((4*x-2)*(4*x-1)))
# (c)
c = integrate_simulate(lambda x: 2*exp(-(log(x)**2)) / x)
# (d)
d = integrate_simulate(lambda x, y: exp((x+y)**2), dimension=2)
# (e)
e = integrate_simulate(lambda x, y: exp(log(x)-y)/x, dimension=2,
                       related=True, relation=lambda x:-log(x))

# problem 5.
EXY = integrate_simulate(lambda x: x*exp(x))
EX = integrate_simulate(lambda x: x)
EY = integrate_simulate(lambda x: exp(x))
cov = EXY - EX * EY

# problem 6.
# (a)
EXY1 = integrate_simulate(lambda x: x*sqrt(1-x**2))
EX1 = integrate_simulate(lambda x: x)
EY1 = integrate_simulate(lambda x: sqrt(1-x**2))
cov1 = EXY1- EX1 * EY1
sigmax1 = sqrt(integrate_simulate(lambda x: x**2) - EX1**2)
sigmay1 = sqrt(integrate_simulate(lambda x: 1-x**2) - EY1**2)
rho1 = cov1 / (sigmax1 * sigmay1)

# (b)
EXY2 = integrate_simulate(lambda x: (x**2)*sqrt(1-x**2))
EX2 = integrate_simulate(lambda x: x**2)
EY2 = integrate_simulate(lambda x: sqrt(1-x**2))
cov2 = EXY2- EX2 * EY2
sigmax2 = sqrt(integrate_simulate(lambda x: x**4) - EX2**2)
sigmay2 = sqrt(integrate_simulate(lambda x: 1-x**2) - EY2**2)
rho2 = cov2 / (sigmax2 * sigmay2)

midsquare_generator()
print('\n')
midsquare_generator(1009, 30)
print('\n')
lcg_generagor()
print('\n')
pseudo_generator()
print('\n')
print('Problem 4, (a):', a)
print('Problem 4, (b):', b)
print('Problem 4, (c):', c)
print('Problem 4, (d):', d)
print('Problem 4, (e):', e)
print('Problem 5, covariance:', cov)
print('Problem 5, (a): rho1:', rho1)
print('Problem 5, (b): rho2:', rho2)
