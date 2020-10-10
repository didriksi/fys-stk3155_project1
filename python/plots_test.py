import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')

def side_by_side(*plots, title="plot", filename="plot", animation=False, init_azim=240):
    """Plots two plots with the same x side by side. Can also make an animation of them from different angles.

    Parameters:
    -----------
    x:          array of shape (n, 2)
                Coordinates of all datapoints. Can be sparse.
    y1:         array of shape (n, )
                Z-value for each pair of coordinates.
    y1_title:   str
                Title to be put above the plot of y1 data.
    y2:         array of shape (n, )
                Z-value for each pair of coordinates.
    y2_title:   str
                Title to be put above the plot of y2 data.
    title:      str
                Title for entire plot.
    filename:   str
                Name of file. Should not include filetype.
    animation:  bool
                Set to true if you want to get a .mp4 of the plots spinning around their own Z-axis.
    init_azim:  float
                Azimuth angle of plot, or initial azimuth angle of animation.
    """
    if len(plots) == 1:
        fig = plt.figure(figsize=(5,5))
        subplot_shape = (1, 1)
    elif len(plots) == 2:
        fig = plt.figure(figsize=(10,5))
        subplot_shape = (1, 2)
    elif len(plots) <= 4:
        fig = plt.figure(figsize=(10,10))
        subplot_shape = (2, 2)
    elif len(plots) <= 6:
        fig = plt.figure(figsize=(10,10))
        subplot_shape = (2, 3)
    elif len(plots) <= 8:
        fig = plt.figure(figsize=(10,12))
        subplot_shape = (2, 4)

    fig.suptitle(title)

    z_lim = (np.min(plots[0][2]), np.max(plots.items()[0][2]))
    axs = []
    for i, (title, (x1, x2), y) in enumerate(plots):
        z_lim = (np.min(np.min(y), z_lim[0]), np.max(np.max(y), z_lim[1]))
        axs.append(fig.add_subplot(*subplot_shape, i+1, projection='3d'))
        axs[i].set_title(title)
        axs[i].plot_trisurf(x1, x2, y, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    for ax in axs:
        axs[i].set_zlim(*zlim)
        axs[i].set_xlabel('x1')
        axs[i].set_ylabel('x2')
        axs[i].view_init(elev=15, azim=init_azim)

    if animation:
        frames = 10
        def frame(i):
            print(f"Drawing frame {i+1}/{frames}")
            axs[0].view_init(elev=15, azim=init_azim+i*36)
            axs[1].view_init(elev=15, azim=init_azim+i*36)
            return fig

        animation = FuncAnimation(fig, frame, frames=range(frames), interval=100)
        animation.save(f"../plots/animations/{filename}.mp4")
    else:
        plt.savefig(f"../plots/{filename}.png")