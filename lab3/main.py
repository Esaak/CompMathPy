import numpy as np
import matplotlib.pyplot as plt


def thomas(a, b, c, d):
    """ A is the tridiagnonal coefficient matrix and d is the RHS matrix"""
    N = len(a)
    cp = np.zeros(N, dtype='float64')  # store tranformed c or c'
    dp = np.zeros(N, dtype='float64')  # store transformed d or d'
    X = np.zeros(N, dtype='float64')  # store unknown coefficients

    # Perform Forward Sweep
    # Equation 1 indexed as 0 in python
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    # Equation 2, ..., N (indexed 1 - N-1 in Python)
    for i in np.arange(1, (N), 1):
        dnum = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / dnum
        dp[i] = (d[i] - a[i] * dp[i - 1]) / dnum

    # Perform Back Substitution
    X[(N - 1)] = dp[N - 1]  # Obtain last xn

    for i in np.arange((N - 2), -1, -1):  # use x[i+1] to obtain x[i]
        X[i] = (dp[i]) - (cp[i]) * (X[i + 1])

    return (X)

class Boundary:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class Solver:
    def __init__(self, scheme, initial, boundary, f, Co, tau, h):
        self.scheme = scheme
        self.state = initial
        self.boundary = boundary
        self.f = f
        self.Co = Co
        self.tau = tau
        self.h = h
        self.t = 0
        self.A = np.ones(len(initial)) * (-Co)
        self.A[0] = 0
        self.A[-1] = -boundary.beta[1] / h

        self.B = np.ones(len(initial)) * (1 + Co) * 2
        self.B[0] = self.boundary.alpha[0] - self.boundary.beta[0] / h
        self.B[-1] = self.boundary.alpha[1] + self.boundary.beta[1] / h

        self.C = np.ones(len(initial)) * (-Co)
        self.C[0] = self.boundary.beta[0]/h
        self.C[-1] = 0

    def solve(self):
        D = []
        D.append(self.boundary.gamma[0])
        # D.append(-self.state[0] * self.h/(2*self.tau) - self.h * self.f(self.scheme[0], self.t + self.tau)/2.)
        #t= []
        for i in range(1, len(self.B) - 1):
            D.append(self.Co * self.state[i + 1] + 2. * (1. - self.Co) * self.state[i] + self.Co * self.state[i - 1] + self.tau * (self.f(self.h * i, self.t + self.tau) + self.f(self.h * i, self.t)) )
            # D.append(self.Co * self.state[i + 1] + 2. * (1. - self.Co) * self.state[i] + self.Co * self.state[i - 1] + self.tau * (self.f(self.t + self.tau) + self.f(self.t)))
            #t.append((self.f(self.h * i, self.t + self.tau) + self.f(self.h * i, self.t)) / 2.)
        D.append(self.boundary.gamma[1])
        # D.append(-self.state[-1] * self.h/(2*self.tau) - self.h * self.f(self.scheme[-1], self.t + self.tau)/2.)
        D = np.array(D)
        # Matrix = np.eye(len(self.B))
        # for j in range(0, len(self.B) -1):
        #     Matrix[j][j] = self.B[j]
        #     Matrix[j][j+1] = self.C[j]
        #     Matrix[j+1][j] = self.A[j]
        # Matrix[-1][-1] = self.B[-1]
        # Matrix[0][0] = -self.boundary.beta[0] * 3/(2. * self.h)
        # Matrix[0][1] = self.boundary.beta[0]*4/(2. * self.h)
        # Matrix[0][2] = - self.boundary.beta[0]/(2.*self.h)
        # Matrix[-1][-1] = self.boundary.beta[-1]* 3 /(2.*self.h)
        # Matrix[-1][-2] = - self.boundary.beta[-1] * 4 /(2.*self.h)
        # Matrix[-1][-3] = self.boundary.beta[-1]/(2*self.h)
        # Matrix[-1][-2] = -1/self.h
        # x_true = np.linalg.solve(Matrix, D)
        x = thomas(self.A, self.B, self.C, D)
        self.state = x
        self.t += self.tau
        return x


def draw(x, y):
    x = np.log(x)
    y = np.log(y)
    plt.figure(figsize=(10, 10))
    plt.xlabel("log(h)")
    plt.ylabel("log(err)")
    coef = np.polyfit(x, y, 1)
    y_approx = coef[0] * np.array(x) + coef[1]
    plt.plot(x, y)
    plt.plot(x, y_approx, "-o", label=f"k = {coef[0]}")
    print(coef[0])
    plt.legend()
    plt.savefig("lab_3_1.jpg")
    plt.show()


if __name__ == '__main__':
    # initial state
    x_0 = 0
    # x_N = np.pi
    x_N = np.pi
    N = np.arange(10, 500, 50)
    T = 10
    tau = 0.01
    # u_0 = 2
    u_0 = 0
    t_N = int(T / tau)
    boundary = Boundary([1., 1.], [0., 0.], gamma=[0, 0])
    # f = lambda x, t: t * np.cos(2 * x)
    # f = lambda x, t: np.exp(-t) * np.cos(5*np.pi * x)
    f = lambda x, t: t * np.sin(2 * x)
    # a = lambda x: 1 / 16. * (np.exp(4 * x)*(4 * x - 1) + 1)
    #f = lambda x, t: 0
    err_N = []
    h_N = []
    for n in N:
        n = int(n)
        initial = np.ones(n) * u_0
        scheme = np.linspace(x_0, x_N, n)
        h = scheme[1] - scheme[0]
        h_N.append(h)
        Co = tau / (h ** 2)

        solver = Solver(scheme, initial, boundary, f, Co, tau, h)
        solution = []
        solution.append(initial)
        # solution
        true_solution = []

        # true_solution.append(initial + 1./(25 * np.pi**2) * (np.exp(-0) - np.exp(-25 * np.pi**2 * 0)) * np.cos(5 * np.pi * scheme) )
        # true_solution.append(initial + a(0) * np.exp(-4 * 0))
        true_solution.append(initial + 1/16. * (np.exp(-4 * 0) + 4*0 - 1) * np.sin(2 * scheme))
        for i in range(1, t_N):
            tmp = solver.solve()
            solution.append(tmp)
            # true_solution.append(initial + a(i * tau) * np.cos(2 * scheme) * np.exp(-4. * i * tau))
            # true_solution.append( initial + 1. / (25 * np.pi ** 2) * (np.exp(-i*tau) - np.exp(-25 * np.pi ** 2 * i*tau)) * np.cos(5 * np.pi * scheme))
            true_solution.append(initial + 1 / 16. * (np.exp(-4 * tau * i) + 4 * tau * i - 1) * np.sin(2 * scheme))
            # true_solution.append(initial + a(i * tau) * np.exp(-4. * i * tau))

        err = []
        err_max = []
        err_max_ind = []
        for i in range(len(true_solution)):
            err.append(np.mean(np.abs(np.array(true_solution[i]) - np.array(solution[i]))))
            temp = np.abs(np.array(true_solution[i]) - np.array(solution[i]))
            err_max.append(np.max(temp))
            err_max_ind.append(np.array(temp).argmax())
        mean_err = np.mean(np.array(err))
        err_N.append(mean_err)
        # print(err_max_ind)
    print(np.log(err_N))
    print(err_N)
    print(np.log(h_N))
    draw(h_N, err_N)
