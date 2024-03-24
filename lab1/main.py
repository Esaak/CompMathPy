import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def corner_func(next, prev, points, Co):
    for i in range(1, len(points)):
        next[i] = prev[i] * (1 - Co) + Co * prev[i - 1]
    next[0] = next[-1]
    return next

def lax_wendroff(next, prev, points, Co):
    for i in range(1, len(points) -1):
        next[i] = Co/2.*(Co - 1) * prev[i+1] + (1 - Co**2) * prev[i] + Co/2. * (Co + 1) * prev[i - 1]
    next[-1] = Co/2.*(Co - 1) * prev[0] + (1 - Co**2) * prev[-1] + Co/2. * (Co + 1) * prev[-2]
    next[0] = Co/2.*(Co - 1) * prev[1] + (1 - Co**2) * prev[0] + Co/2. * (Co + 1) * prev[-1]
    return next

def draw(time_steps, answers, points, path, title, sign):
    plt.figure(figsize=(10, 10))

    for i in range(len(time_steps) - 1):
        plt.clf()
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        answers = np.array(answers)
        plt.ylim([np.min(answers), np.max(answers)])
        plt.title(title)
        plt.scatter(points, answers[i], label=sign + f" time = {np.round(time_steps[i], 2)}")
        plt.legend(loc='upper right')
        plt.savefig(path + str(i) + ".png")

    frames = [Image.open(image) for image in glob.glob(f"{path}/*.png")]
    frame_one = frames[0]
    frame_one.save(path + "my_awesome.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)




if __name__ == '__main__':

    # initial conditions
    h = 0.5
    T = 18.0
    L = 20.0
    answers = []
    points = np.arange(0, L + h, h)
    CFL = 1.1
    tau = CFL * h
    time_steps = np.arange(0, T + tau, tau)
    #y_prev = np.sin(4 * np.pi * points / L) #sin(4pi*x/L)
    init = 0
    zero = 10.
    y_prev = []
    # for point in points:
    #     if point < zero  + L/6.:
    #         y_prev.append(np.exp(-(L/6.)**2/(1e-12 + (L/6.)**2 - (point - zero)**2)))
    #     else:
    #         zero += L/3.
    #         y_prev.append(np.exp(-(L/6.)**2/(1e-12 + (L/6.)**2 - (point - zero)**2)))
    for point in points:
        if point < 8. or point > 12.:
            y_prev.append(0.)
        else:
            y_prev.append(np.exp(-(2) ** 2 / (1e-12 + 2 ** 2 - (point - zero) ** 2)))
    y_next = np.ones(len(points))
    answers.append(y_prev)
    print(time_steps[-1])
    for _ in time_steps[: -2]:
        #y_next = np.copy(lax_wendroff(y_next, y_prev, points, CFL))
        y_next = np.copy(corner_func(y_next, y_prev, points, CFL))
        answers.append(np.copy(y_next))
        y_prev = np.copy(y_next)
    draw(time_steps, answers, points, "./images/corner_func11_shapochka/", "Лабораторная работа 1. Левый уголок", f'Co = {CFL}')
    #draw(time_steps, answers, points, "./images/lax_wendroff06_shapochka/", "Лабораторная работа 1. Лакс-Вендрофф", f'Co = {CFL}')
