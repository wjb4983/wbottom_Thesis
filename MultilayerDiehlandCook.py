from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection

class MultiLayerDiehlAndCook2015(Network):
    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        num_layers: int = 10,
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
    ) -> None:
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.num_layers = num_layers
        self.exc = exc
        self.inh = inh
        self.dt = dt
        
        # Layers and connections lists
        self.input_layers = []
        self.exc_layers = []
        self.inh_layers = []
        self.input_exc_connections = []
        self.exc_inh_connections = []
        self.inh_exc_connections = []
        layer_names = []
        
        # Create input layer
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.input_layers.append(input_layer)
        layer_names.append("X")
        self.add_layer(input_layer, name="X")
        
        
        for i in range(self.num_layers):
            # Create excitatory layer
            # exc_layer = DiehlAndCookNodes(
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
            exc_layer = LIFNodes(
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
            self.exc_layers.append(exc_layer)
            layer_names.append(f"Ae_{i}")
            self.add_layer(exc_layer, name=f"Ae_{i}")

            # Create connections
            if i == 0:  # If it's the first layer
                w_input_exc = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
                source_layer = "X"  # Source is the input layer
            else:
                w_input_exc = 0.3 * torch.rand(self.n_neurons, self.n_neurons)
                source_layer = f"Ae_{i-1}"  # Source is the previous excitatory layer
            
            input_exc_conn = Connection(
                source=self.layers[source_layer],  # Use the appropriate source layer
                target=exc_layer,
                w=w_input_exc,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
                )
            self.input_exc_connections.append(input_exc_conn)
            if(i == num_layers-1):
                # Create inhibitory layer
                inh_layer = LIFNodes(
                    n=self.n_neurons,
                    traces=False,
                    rest=-60.0,
                    reset=-45.0,
                    thresh=inh_thresh,
                    tc_decay=10.0,
                    refrac=2,
                    tc_trace=20.0,
                )
                self.inh_layers.append(inh_layer)
                layer_names.append(f"Ai_{i}")
                self.add_layer(inh_layer, name=f"Ai_{i}")
            
                w_exc_inh = self.exc * torch.diag(torch.ones(self.n_neurons))
                exc_inh_conn = Connection(
                    source=exc_layer, target=inh_layer, w=w_exc_inh, wmin=0, wmax=self.exc
                )
                self.exc_inh_connections.append(exc_inh_conn)
            
                w_inh_exc = -self.inh * (
                    torch.ones(self.n_neurons, self.n_neurons)
                    - torch.diag(torch.ones(self.n_neurons))
                )
                inh_exc_conn = Connection(
                    source=inh_layer, target=exc_layer, w=w_inh_exc, wmin=-self.inh, wmax=0
                )
                self.inh_exc_connections.append(inh_exc_conn)
                self.add_connection(exc_inh_conn, source=f"Ae_{i}", target=f"Ai_{i}")
                self.add_connection(inh_exc_conn, source=f"Ai_{i}", target=f"Ae_{i}")
            if(i==0):
                self.add_connection(input_exc_conn, source="X", target=f"Ae_{i}")
            else:
                self.add_connection(input_exc_conn, source=f"Ae_{i-1}", target=f"Ae_{i}")
        
            # Set the next input layer to the current excitatory layer
            input_layer = exc_layer