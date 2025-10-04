import matplotlib.pyplot as plt
import numpy as np

def radar_chart(ax, data):
    # data is transposed
    values = data.values
    indices = data.index
    N = len(indices)

    values = np.append(values, values[0])
    angles = np.arange(N + 1) / float(N) * 2 * np.pi

    plt.xticks(angles[:-1], indices)
    ax.yaxis.get_major_locator().base.set_params(nbins=3)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    plt.title(data.name)