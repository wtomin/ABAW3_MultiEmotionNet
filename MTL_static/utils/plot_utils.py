from matplotlib import pyplot as plt
import matplotlib
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
import numpy as np
def plot_distribution(au_array, au_ids_list, au_names_list, title):
    width = 0.1
    ys = []
    for i_au in range(len(au_ids_list)):
        au_intensity = au_array[:, i_au]
        num_samples = [sum(au_intensity==i) for i in range(6)]
        ys.append(num_samples)
    ys = np.array(ys) # (12, 5)
    for i in range(6):
        offset = (3 - i)*width
        plt.bar(np.arange(len(au_ids_list)) - offset, ys[:, i], label="{:d}".format(i), width=width)

    plt.legend(title='intensity')
    plt.xlabel("Action Units")
    plt.ylabel("#. samples")
    plt.xticks(np.arange(len(au_ids_list)), au_names_list)
    if not title is None:
        plt.title(title)
    plt.show()

