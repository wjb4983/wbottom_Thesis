from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre, MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes, CurrentLIFNodes
from bindsnet.network.topology import Connection, LocalConnection

class MultiLayerMSTDPET(Network):

    def __init__(

        self,
        n_inpt: int,
        n_neurons: int = 100,
        num_layers: int = 1,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        inh_thresh: float = -40.0,
        exc_thresh: float = -52.0,
        redundancy: bool = False,
        last_layer_inh: bool = False,
        seed: int = 0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer_1 = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        w = 0.7*2 * torch.rand(self.n_inpt, self.n_neurons)
        input_middle = Connection(
            source=input_layer,
            target=exc_layer_1,
            w=w,
            update_rule=MSTDPET,
            nu=(1e-8,1e-2),
            reduction=reduction,
            wmin=wmin,
            wmax=wmax*1.5,
            # norm=1,
            )
        input_middle.update_rule.reduction = torch.sum
        self.add_layer(input_layer, "X")
        self.add_layer(exc_layer_1, f"Ae_0")
        self.add_connection(input_middle, source="X", target=f"Ae_0")
        exc_layer_2 = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0, 
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            )
        w = 1.76*2 * torch.rand(self.n_neurons, self.n_neurons)
        middle_end = Connection(
            source=exc_layer_1,
            target=exc_layer_2,
            w=w,
            update_rule=MSTDPET,
            nu=(1e-9,1e-2),
            reduction=reduction,
            wmin=wmin,
            wmax=wmax*2.5,
            # norm=norm,
            )
        middle_end.update_rule.reduction = torch.sum
        self.add_layer(exc_layer_2, f"Ae_1")
        self.add_connection(middle_end, source=f"Ae_0", target=f"Ae_1")

        # exc_layer_1_1 = DiehlAndCookNodes(
        #     n=self.n_neurons,
        #     traces=True,
        #     rest=-65.0,
        #     reset=-60.0,
        #     thresh=exc_thresh,
        #     refrac=5,
        #     tc_decay=100.0,
        #     tc_trace=20.0,
        #     theta_plus=theta_plus,
        #     tc_theta_decay=tc_theta_decay,
        # )
        # w = 0.7*2  * torch.rand(self.n_inpt, self.n_neurons)
        # input_middle_1 = Connection(
        #     source=input_layer,
        #     target=exc_layer_1,
        #     w=w,
        #     update_rule=MSTDPET,
        #     nu=(1e-8,1e-2),
        #     reduction=reduction,
        #     wmin=wmin,
        #     wmax=wmax*1.5,
        #     # norm=1,
        #     )
        # input_middle_1.update_rule.reduction = torch.sum
        # self.add_layer(input_layer, "X")
        # self.add_layer(exc_layer_1_1, f"Ae_0_0")
        # self.add_connection(input_middle_1, source="X", target=f"Ae_0_0")
        
        # w = 3.0 * torch.rand(self.n_neurons, self.n_neurons)
        # middle_end_1 = Connection(
        #     source=exc_layer_1_1,
        #     target=exc_layer_2,
        #     w=w,
        #     update_rule=MSTDPET,
        #     nu=(1e-9,1e-2),
        #     reduction=reduction,
        #     wmin=wmin,
        #     wmax=wmax*2.5,
        #     # norm=norm,
        #     )
        # middle_end_1.update_rule.reduction = torch.sum
        # self.add_connection(middle_end_1, source=f"Ae_0_0", target=f"Ae_1")
       