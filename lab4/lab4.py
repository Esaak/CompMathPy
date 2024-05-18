import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def conjgrad(A, b, x, dis):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    r = b - A.dot(x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(len(b)):
        Ap = A.dot(p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < dis:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

class Boundary:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class Solver:
    def __init__(self, scheme_x, scheme_y, initial, boundary, f, Co, tau, h):
        self.scheme_x = scheme_x
        self.scheme_y = scheme_y
        self.state = initial
        self.boundary = boundary
        self.f = f
        self.Co = Co
        self.tau = tau
        self.h = h
        self.t = tau
        self.N = len(scheme_y)
        # self.Matrix_2d = self.create_matrix(Co, self.N)
        # self.CSR = sp.sparse.csr_matrix(self.Matrix_2d)
        self.CSR = self.create_matrix(Co, self.N)
        self.M = self.CSR.toarray()

    def create_matrix(self, Co, N):
        row = []
        col = []
        data = []
        # Matrix = np.eye(N * N)
        # Matrix[N: N*(N-1), N: N*(N-1)] *= (1 + 26. * Co)
        for i in range(N):
            row.append(i)
            col.append(i)
            data.append(1.)
            row.append(i + N * (N - 1))
            col.append(i + N * (N - 1))
            data.append(1.)
        for i in range(N, N*(N-1)):
            row.append(i)
            col.append(i)
            data.append(1 + 26. * Co)
        for i in range(1, N -1):
            for j in range(N*i, N*(i+1)):
                if j + 1 != (i + 1) * N and j + 1 < N*N:
                    if (j + 1) % N + 1 != N:
                        row.append(j + 1)
                        col.append(j)
                        data.append(-Co / 2.)
                        # Matrix[j + 1, j] = -  Co / 2.
                    if j % N !=0:
                        row.append(j)
                        col.append(j+1)
                        data.append(-Co / 2.)
                        # Matrix[j, j + 1] = -  Co / 2.
                if j+N < N*N and j % N != 0 and j % N + 1 != N:
                    row.append(j)
                    col.append(j -N)
                    data.append(- 25 *Co / 2.)
                    row.append(j)
                    col.append(j + N)
                    data.append(- 25 * Co / 2.)
                    # Matrix[j, j -N] = - 25 *Co / 2.
                    # Matrix[j, j + N] = -25 *Co / 2.

        return sp.sparse.csr_matrix((np.array(data),(np.array(row, dtype=int), np.array(col, dtype=int))), shape=(N*N, N*N))

    def solve(self):
        D = np.ones(self.N * self.N)
        U = np.ones(self.N * self.N)
        # D.append(self.boundary.gamma[0])
        # t= []
        for i in range(self.N):
            D[i*self.N] = 0
            D[i*self.N + self.N -1] = 0
            D[i] = self.boundary.gamma[0][0](i*h, self.t)
            D[i + self.N * (self.N - 1)] = self.boundary.gamma[0][1](i*h, self.t)
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                D[j + i*self.N] =(1. - 26. * self.Co) * self.state[i, j] + self.Co / 2. * ((self.state[i + 1, j] + self.state[i - 1, j]) * 25. +
                                                                           self.state[i, j + 1] + self.state[i, j - 1] +
                                                                           self.tau / 2. * (f(h * i, h * j, self.t) +
                                                                                            f(h * i, h * j, self.t + tau)))

        for i in range(self.N):
            for j in range(self.N):
                U[j + i * self.N] = self.state[i, j]
        D = np.array(D)
        r = self.h ** 3
        # x = np.linalg.solve(self.Matrix_2d,D)
        x = conjgrad(self.CSR, D, U, r)
        for i in range(self.N):
            for j in range(self.N):
                self.state[i, j] = x[j + i * self.N]
        self.t += self.tau
        return self.state


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
    x_N = 1.
    y_0 = 0
    y_N = 1.
    N = np.arange(10, 130, 30)
    T = 3
    tau = 0.01
    u_0 = 0
    t_N = int(T / tau)
    lmbda = 1e-4
    b_x_d = lambda y, t: np.sin(5 * np.pi * y) * np.exp(-50 * np.pi ** 2 * lmbda * t)
    b_x_u = lambda y, t: -np.sin(5 * np.pi * y) * np.exp(-50 * np.pi ** 2 * lmbda * t)
    b_y_d = lambda y, t: 0.
    b_y_u = lambda y, t: 0.
    initial_func = lambda x, y: np.cos(np.pi * x) * np.sin(5 * np.pi * y)
    boundary = Boundary([[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]], gamma=[[b_x_d, b_x_u], [b_y_d, b_y_u]])

    true_sol_func = lambda x, y, t: np.cos(np.pi * x) * np.sin(5 * np.pi * y) * np.exp(-50 * np.pi ** 2 * lmbda * t)
    f = lambda x, y, t: 0.
    err_N = []
    h_N = []
    for n in N:
        scheme_x = np.linspace(x_0, x_N, n)
        scheme_y = np.linspace(y_0, y_N, n)
        h = scheme_x[1] - scheme_x[0]

        n = int(n)
        initial = np.eye(n)
        for i in range(n):
            for j in range(n):
                initial[i][j] = initial_func(i * h, j * h)

        h_N.append(h)
        Co = tau * lmbda / (h ** 2)

        solver = Solver(scheme_x, scheme_y, initial, boundary, f, Co, tau, h)
        solution = []
        solution.append(np.copy(initial))
        # solution
        true_solution = []

        tmp = np.ones((n,n))
        true_solution.append(np.copy(initial))
        err = []
        for te in range(1, t_N):
            temp = np.copy(solver.solve())
            # solution.append(np.copy(temp))
            tmp = np.ones((n,n))
            for m in range(n):
                for k in range(n):
                    tmp[m][k] = true_sol_func(h * m, h * k, tau * te)
            # true_solution.append(np.copy(tmp))
            err.append(np.mean(np.abs(np.copy(tmp) - np.copy(temp))))
        err_max = []
        err_max_ind = []
        mean_err = np.mean(np.array(err))
        err_N.append(mean_err)
        print(n, err_N)
    print(np.log(err_N))
    print(err_N)
    print(np.log(h_N))
    draw(h_N, err_N)
