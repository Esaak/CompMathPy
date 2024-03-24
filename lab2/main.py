import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def KIR(next, prev, A, OmegaT, OmegaT_1, Ld, Co):
    B = OmegaT_1 * Ld * OmegaT
    for i in range(1, len(next) - 1):
        next[i] = prev[i] - Co / 2. * A * (prev[i + 1] - prev[i - 1]) + Co / 2. * B * (
                prev[i + 1] - 2 * prev[i] + prev[i - 1])
    next[-1] = next[-2]
    next[0] = next[1]
    return next


def update_matrix(eps, u, c, gamma=5 / 3.):
    OmegaT = [[-u * c, c, gamma - 1.], [-c * c, 0, gamma - 1.], [u * c, -c, gamma - 1]]
    OmegaT_1 = [[1 / (2. * c ** 2), -1. / c ** 2, 1 / (2. * c ** 2)],
                [(u + c) / (2. * c ** 2), -u / (c ** 2), (u - c) / (2 * c ** 2)],
                [1 / (2 * (gamma - 1)), 0, 1 / (2. * (gamma - 1))]]
    A = [[0, 1., 0.], [-u ** 2, 2 * u, gamma - 1], [-gamma * u * eps, gamma * eps, u]]
    Ld = [[np.abs(u + c), 0, 0], [0, np.abs(u), 0], [0, 0, np.abs(u - c)]]
    return OmegaT, OmegaT_1, A, Ld


def draw(time_steps, answers, points, path, title):
    plt.figure(figsize=(10, 10))

    for i in range(len(time_steps) - 1):
        if i % 100 == 0:
            plt.clf()
            plt.grid()
            plt.xlabel("x")
            plt.ylabel("y")
            answers = np.array(answers)
            plt.ylim([np.min(answers), np.max(answers)])
            plt.title(title)
            plt.scatter(points, answers[i], label=f" time = {np.round(time_steps[i], 6)}")
            plt.legend(loc='upper right')
            plt.savefig(path + str(i) + ".png")

    frames = [Image.open(image) for image in glob.glob(f"{path}/*.png")]
    frame_one = frames[0]
    frame_one.save(path + "my_awesome.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


if __name__ == '__main__':

    # initial conditions
    h = 0.1
    CFL = 0.008
    T = 0.02
    cur_time = 0.
    L = 10.0
    gamma = 5 / 3.
    points = np.arange(-L, L + h, h)
    if points[-1] > L:
        points = points[:-2]

    # left
    vL = 0.
    roL = 13.
    pL = 10. * 10 ** 5
    # right
    vR = 0.
    roR = 1.3
    pR = 1. * 10 ** 5
    u = 0.
    eps = 0.
    epsL = pL / ((gamma - 1) * roL)
    epsR = pR / ((gamma - 1) * roR)
    cL = np.sqrt(gamma * (gamma - 1) * epsL)
    cR = np.sqrt(gamma * (gamma - 1) * epsR)
    time_steps = [0.]
    # НУ
    w_prev = []
    nu = []
    nu_results = []
    p_results = []
    p = []
    OmegaT_prev = []
    OmegaT_1_prev = []
    A_prev = []
    Ld_prev = []
    for i in range(len(points)):
        if points[i] <= 0:
            w_prev.append([roL, 0., roL * epsL])
            nu.append([roL, 0, epsL])
            p.append(pL)
            O, O_1, A, Ld = update_matrix(epsL, u, cL)
            OmegaT_prev.append(np.array(O))
            OmegaT_1_prev.append(np.array(O_1))
            A_prev.append(np.array(A))
            Ld_prev.append(np.array(Ld))
        else:
            w_prev.append([roR, 0., roR * epsR])
            nu.append([roR, 0, epsR])
            p.append(pR)
            O, O_1, A, Ld = update_matrix(epsR, u, cR)
            OmegaT_prev.append(np.array(O))
            OmegaT_1_prev.append(np.array(O_1))
            A_prev.append(np.array(A))
            Ld_prev.append(np.array(Ld))
    w_next = np.copy(w_prev)
    nu_results.append(np.copy(nu))
    p_results.append(np.copy(p))
    w_prev = np.array(w_prev)
    w_next = np.array(w_next)
    while cur_time < T:
        tau = CFL * h / np.max(Ld_prev)
        Co = tau / h
        for i in range(1, len(points) - 1):
            B = np.matmul(OmegaT_1_prev[i], Ld_prev[i])
            B = np.matmul(B, OmegaT_prev[i])
            w_next[i] = w_prev[i] - Co / 2. * np.matmul(A_prev[i], (w_prev[i + 1] - w_prev[i - 1])) + Co / 2. * np.matmul(B, (
                    w_prev[i + 1] - 2 * w_prev[i] + w_prev[i - 1]))
            nu[i] = [w_next[i][0], w_next[i][1] / w_next[i][0], w_next[i][2] / w_next[i][0]]
            p[i] = w_next[i][2] * (gamma - 1)
            c = np.sqrt(nu[i][2] * gamma * (gamma - 1))
            O, O_1, A, Ld = update_matrix(nu[i][2], nu[i][1], c)
            if nu[i][2] < 0:
                print(cur_time, i, len(time_steps))
                break
            OmegaT_prev[i] = np.array(O)
            OmegaT_1_prev[i] = np.array(O_1)
            A_prev[i] = np.array(A)
            Ld_prev[i] = np.array(Ld)

        w_next[0] = w_next[1]
        w_next[-1] = w_next[-2]
        w_prev = np.copy(w_next)

        nu[0] = nu[1]
        nu[-1] = nu[-2]
        OmegaT_prev[0] = OmegaT_prev[1]
        OmegaT_1_prev[0] = OmegaT_1_prev[1]
        A_prev[0] = A_prev[1]
        Ld_prev[0] = Ld_prev[1]

        OmegaT_prev[-1] = OmegaT_prev[-2]
        OmegaT_1_prev[-1] = OmegaT_1_prev[-2]
        A_prev[-1] = A_prev[-2]
        Ld_prev[-1] = Ld_prev[-2]

        p[0] = p[1]
        p[-1] = p[-2]

        nu_results.append(np.copy(nu))
        p_results.append(np.copy(p))
        #tmp = np.array(nu_results[-1]) - np.array(nu_results[0])
        cur_time += tau
        time_steps.append(cur_time)
    nu_results = np.array(nu_results)
    print(len(nu_results[0]), len(points))
    Ro = []
    U = []
    E = []
    for i in range(len(nu_results)):
        Ro.append(nu_results[i][:, 0])
        U.append(nu_results[i][:, 1])
        E.append(nu_results[i][:, 2])
    draw(time_steps, p_results, points, "./images/p/", "Лабораторная работа 2. P, Па")
    draw(time_steps, Ro, points, "./images/nu1/", "Лабораторная работа 2. U, m/s")
    draw(time_steps, U, points, "./images/nu0/", "Лабораторная работа 2. Ro, kg/m^3")
    draw(time_steps, E, points, "./images/nu2/", "Лабораторная работа 2. e, Dg/kg")