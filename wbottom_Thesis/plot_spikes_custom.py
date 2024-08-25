from typing import Dict, List, Optional, Sized, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.modules.utils import _pair
from matplotlib.figure import Figure

from bindsnet.utils import (
    reshape_conv2d_weights,
    reshape_locally_connected_weights,
    reshape_local_connection_2d_weights,
)
def plot_spikes_custom(
    spikes: Dict[str, torch.Tensor],
    time: Optional[Tuple[int, int]] = None,
    n_neurons: Optional[Dict[str, Tuple[int, int]]] = None,
    ims: Optional[List[PathCollection]] = None,
    axes: Optional[Union[Axes, List[Axes]]] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
# ) -> Tuple[List[AxesImage], List[Axes]]:
    plots_per_figure: int = 2,
    fig: Optional[Figure] = None
    )-> Tuple[List[PathCollection], List[Axes], Figure]:

    """
    Plot spikes for any group(s) of neurons.

    :param spikes: Mapping from layer names to spiking data.
    :param time: Plot spiking activity of neurons in the given time range.
    :param n_neurons: Plot spiking activity of neurons in the given range of neurons.
    :param figsize: Horizontal, vertical figure size in inches.
    :param plots_per_figure: Number of plots to include in each figure.
    :return: Tuple containing the list of ims (scatter plot objects) and axes.
    """
    if n_neurons is None:
       n_neurons = {}

    spikes = {k: v.view(v.size(0), -1) for k, v in spikes.items()}
    if time is None:
        for key in spikes.keys():
            time = (0, spikes[key].shape[0])
            break
 
    # if axes is None:# or fig is None:
    #     x, axes = plt.subplots(plots_per_figure, 1, figsize=figsize) #plt, 
    #     axes = np.array(axes).flatten()
    for key, val in spikes.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.shape[1])
 
    ims_list = []
    fig_list = fig
    if(fig_list is None):
        fig_list = []
    axes_list = []
    print(fig_list)
    
    # Iterate over layers and create plots
    for layer_idx in range(0, len(spikes), plots_per_figure):
        # print(layer_idx)
        if fig_list is None or len(fig_list) <= layer_idx/2:
                fig_ = plt.figure(figsize=figsize)
                fig_.subplots_adjust(hspace=0.5)
                fig_list.append(fig_)
        fig = fig_list[int(layer_idx/2)]
        ax1 = fig.add_subplot(plots_per_figure, 1, 1)
        ax2 = fig.add_subplot(plots_per_figure, 1, 2)
        plot_idx = layer_idx + len(axes_list)
        if plot_idx >= len(spikes):
            break

        layer_name = list(spikes.keys())[plot_idx]
        spike_data = spikes[layer_name]
        if layer_name not in n_neurons:
            n_neurons[layer_name] = (0, spike_data.shape[1])

        spikes_data = spike_data[
            time[0]:time[1], n_neurons[layer_name][0]:n_neurons[layer_name][1]
        ].detach().clone().cpu().numpy()

        ims = ax1.scatter(
            x=np.array(spikes_data.nonzero()).T[:, 0],
            y=np.array(spikes_data.nonzero()).T[:, 1],
            s=1,
        )
        args = (
            layer_name,
            n_neurons[layer_name][0],
            n_neurons[layer_name][1],
            time[0],
            time[1],
        )
        ax1.set_title("%s spikes for neurons (%d - %d) from t = %d to %d " % args)
        ax1.set_yticks([n_neurons[layer_name][0], n_neurons[layer_name][1]])
        ax1.set_aspect("auto")
        ims_list.append(ims)
        axes_list.append(ax1)

        # Second subplot
        ims = ax2.scatter(
            x=np.array(spikes_data.nonzero()).T[:, 0],
            y=np.array(spikes_data.nonzero()).T[:, 1],
            s=1,
        )
        ax2.set_title("%s spikes for neurons (%d - %d) from t = %d to %d " % args)
        ax2.set_yticks([n_neurons[layer_name][0], n_neurons[layer_name][1]])
        ax2.set_aspect("auto")
        ims_list.append(ims)
        axes_list.append(ax2)

    plt.setp([ax1, ax2], xticks=[], xlabel="Simulation time", ylabel="Neuron index")
    plt.tight_layout()

    return ims_list, axes_list, fig