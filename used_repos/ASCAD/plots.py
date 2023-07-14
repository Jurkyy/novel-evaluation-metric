import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.arange(10, 1000, 10)

    prefix = 'succ_rate_'
    for filename in os.listdir('plot_data/hw_data'):
        if filename.startswith(prefix):
            y = np.load('plot_data/hw_data/' + filename)
            plt.plot(x, y, label= filename.replace(".npy", "").replace(prefix, ""))

    plt.legend()
    plt.show()