from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre, WeightDependentPostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes, CurrentLIFNodes
from bindsnet.network.topology import Connection, LocalConnection

class MultiLayerDiehlAndCook2015(Network):

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
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_middle = Connection(
            source=input_layer,
            target=exc_layer_1,
            w=w,
            update_rule=PostPre,
            nu=(1e-4,1e-2),
            reduction=reduction,
            wmin=wmin,
            wmax=exc*3,
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
        w = 0.3 * torch.rand(self.n_neurons, self.n_neurons)
        nu = (1e-4, 1e-2)
        middle_end = Connection(
            source=exc_layer_1,
            target=exc_layer_2,
            w=w,
            update_rule=PostPre,
            nu=(1e-4,1e-2),
            reduction=reduction,
            wmin=wmin,
            wmax=exc,
            # norm=norm,
            )
        middle_end.update_rule.reduction = torch.sum
        self.add_layer(exc_layer_2, f"Ae_1")
        self.add_connection(middle_end, source=f"Ae_0", target=f"Ae_1")
        # for i in range(num_layers):
        #     if(i == num_layers - 1):
        #         exc_layer = DiehlAndCookNodes(
        #             n=self.n_neurons,
        #             traces=True,
        #             rest=-65.0,
        #             reset=-65.0,
        #             thresh=exc_thresh,
        #             refrac=5,
        #             tc_decay=100.0,
        #             tc_trace=20.0, 
        #             theta_plus=theta_plus,
        #             tc_theta_decay=tc_theta_decay,
        #         )
        #         w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        #         nu = (1e-4, 1e-3)
        #         input_last_conn = Connection(
        #             source=il,
        #             target=exc_layer,
        #             w=w,
        #             update_rule=PostPre,
        #             nu=nu,
        #             reduction=reduction,
        #             wmin=wmin,
        #             wmax=wmax,
        #             norm=norm,
        #             )
        #         input_last_conn.update_rule.reduction = torch.sum
        #         self.add_connection(input_last_conn, source="X", target=f"Ai_{i}")
        #         if(last_layer_inh):
        #             inh_layer = LIFNodes(
        #                 n=self.n_neurons,
        #                 traces=False,
        #                 rest=-65.0,
        #                 reset=-60.0,
        #                 thresh=inh_thresh,
        #                 tc_decay=10.0,
        #                 refrac=2,
        #                 tc_trace=20.0,
        #             )
        #             w = self.exc * torch.diag(torch.ones(self.n_neurons))
        #             exc_inh_conn = Connection(
        #                 source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        #             )
        #             w = -self.inh * (
        #                 torch.ones(self.n_neurons, self.n_neurons)
        #                 - torch.diag(torch.ones(self.n_neurons))
        #             )
        #             inh_exc_conn = Connection(
        #                 source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        #             )
        #             self.add_layer(inh_layer, name=f"Ai_{i}")
    
        #             self.add_connection(exc_inh_conn, source=f"Ae_{i}", target=f"Ai_{i}")
        #             self.add_connection(inh_exc_conn, source=f"Ai_{i}", target=f"Ae_{i}")
        #     else:
        #         exc_layer = DiehlAndCookNodes(
        #             n=self.n_neurons,
        #             traces=True,
        #             rest=-65.0,
        #             reset=-60.0,
        #             thresh=exc_thresh,
        #             refrac=5,
        #             tc_decay=100.0,
        #             tc_trace=20.0,
        #             theta_plus=theta_plus,
        #             tc_theta_decay=tc_theta_decay,
        #         )
        #     if(i==0 and redundancy):
        #         ##################################################
        #         #Connection from X to L1 redundancy
        #         #Adds 2 layers for redundancy - regular LIF and current based LIF
        #         #creates input conneciton here but need to create layer 2 connection later
        #         # exc_layer_redun1 = CurrentLIFNodes(
        #         #     n=self.n_neurons,
        #         #     traces=True,
        #         #     rest=-55.0,
        #         #     reset=-80.0,
        #         #     thresh=exc_thresh,  # Use the same threshold as in DiehlAndCookNodes
        #         #     refrac=5.0,
        #         #     tc_decay=100.0,
        #         #     tc_trace=2.0,
        #         #     one_spike = True,
        #         # )
        #         exc_layer_redun1 = DiehlAndCookNodes(
        #             n=self.n_neurons,
        #             traces=True,
        #             rest=-65.0,
        #             reset=-60.0,
        #             thresh=exc_thresh,
        #             refrac=5,
        #             tc_decay=100.0,
        #             tc_trace=20.0, #increased to be more sensitive
        #             theta_plus=theta_plus,
        #             tc_theta_decay=tc_theta_decay,
        #         )
        #         self.add_layer(exc_layer_redun1, name=f"Ae_{i}_r1")
        #         # exc_layer_redun2 = LIFNodes(
        #         #     n=self.n_neurons,
        #         #     traces=True,
        #         #     rest=-65.0,
        #         #     reset=-60.0,
        #         #     thresh=exc_thresh,
        #         #     refrac=5,
        #         #     tc_decay=100.0,
        #         #     tc_trace=20.0,
        #         #     theta_plus=theta_plus,
        #         #     tc_theta_decay=tc_theta_decay,
        #         # )
        #         # self.add_layer(exc_layer_redun2, name=f"Ae_{i}_r2")
        #         w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        #         input_exc_conn_r1 = Connection(
        #             source=input_layer,
        #             target=exc_layer_redun1,
        #             w=w,
        #             update_rule=PostPre,
        #             nu=nu,
        #             reduction=reduction,
        #             wmin=wmin,
        #             wmax=wmax,
        #             norm=norm,
        #         )
        #         # w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        #         # input_exc_conn_r2 = Connection(
        #         #     source=input_layer,
        #         #     target=exc_layer_redun2,
        #         #     w=w,
        #         #     update_rule=PostPre,
        #         #     nu=nu,
        #         #     reduction=reduction,
        #         #     wmin=wmin,
        #         #     wmax=wmax,
        #         #     norm=norm/4,
        #         # )
        #         self.add_connection(input_exc_conn_r1, source="X", target=f"Ae_{i}_r1")
        #         # self.add_connection(input_exc_conn_r2, source="X", target=f"Ae_{i}_r2")
                
        #         # inh_layer_r1 = LIFNodes(
        #         #     n=self.n_neurons,
        #         #     traces=False,
        #         #     rest=-60.0,
        #         #     reset=-30.0,
        #         #     thresh=inh_thresh,
        #         #     tc_decay=10.0,
        #         #     refrac=2,
        #         #     tc_trace=20.0,
        #         # )
        #         # w = self.exc * torch.diag(torch.ones(self.n_neurons))
        #         # exc_inh_conn_r1 = Connection(
        #         #     source=exc_layer_redun1, target=inh_layer_r1, w=w, wmin=0, wmax=self.exc
        #         # )
        #         # w = -self.inh * (
        #         #     torch.ones(self.n_neurons, self.n_neurons)
        #         #     - torch.diag(torch.ones(self.n_neurons))
        #         # )
        #         # inh_exc_conn_r1 = Connection(
        #         #     source=inh_layer_r1, target=exc_layer_redun1, w=w, wmin=-self.inh, wmax=0
        #         # )
        #         # self.add_layer(inh_layer_r1, name=f"Ai_{i}_r1")

        #         # self.add_connection(exc_inh_conn_r1, source=f"Ae_{i}_r1", target=f"Ai_{i}_r1")
        #         # self.add_connection(inh_exc_conn_r1, source=f"Ai_{i}_r1", target=f"Ae_{i}_r1")
        #         ##################################################
        #         ##################################################
        #         ##################################################
        #     # Connections fom Previous layer to current layer
        #     #For 2 layer model, i=0 => X->L1, i=1 L1->L2
        #     #Need to specift input size for input layer since 1,28,28 image
        #     if(i==0):
        #         nu = (1e-4, 1e-3)#1e-3)
        #         # nu = (1e-3, 1e-3)
        #     else:
        #         nu = (0, 1e-2)
        #         # nu = (1e-3, 1e-3)
        #     if(i==0):
        #         w = exc * torch.rand(self.n_inpt, self.n_neurons)
        #     else:
        #         w = exc * torch.rand(self.n_neurons, self.n_neurons)
        #     #Norm = 1 to make normalization 0
        #     #try to increase wmin
        #     input_exc_conn = Connection(
        #         source=input_layer,
        #         target=exc_layer,
        #         w=w,
        #         update_rule=PostPre,
        #         nu=nu,
        #         reduction=reduction,
        #         wmin=wmin,
        #         wmax=exc,
        #         norm=1,
        #     )
        #     input_exc_conn.update_rule.reduction = torch.sum

    
        #     # Where we self add connections
        #     ##################################################
        #     if(i==0):
        #         self.add_layer(input_layer, name="X")
        #         self.add_connection(input_exc_conn, source="X", target=f"Ae_{i}")
        #     else:
        #         self.add_layer(input_layer, name=f"Ae_{i-1}")
        #         self.add_connection(input_exc_conn, source=f"Ae_{i-1}", target=f"Ae_{i}")
        #     self.add_layer(exc_layer, name=f"Ae_{i}")
        #     ##################################################
        #     #For the redundancy
        #     if(i==1 and redundancy):
        #         w = exc * torch.rand(self.n_neurons, self.n_neurons)
        #         redun1_layer2 = Connection(
        #             source=input_layer,
        #             target=exc_layer,
        #             w=w,
        #             update_rule=PostPre,
        #             nu=nu,
        #             reduction=reduction,
        #             wmin=wmin,
        #             wmax=0,#exc,
        #             norm=norm,
        #         )
        #         # w = 0.3 * torch.rand(self.n_neurons, self.n_neurons)
        #         # redun2_layer2 = Connection(
        #         #     source=input_layer,
        #         #     target=exc_layer,
        #         #     w=w,
        #         #     update_rule=PostPre,
        #         #     nu=nu,
        #         #     reduction=reduction,
        #         #     wmin=wmin,
        #         #     wmax=wmax,
        #         #     norm=norm,
        #         # )
        #         redun1_layer2.update_rule.reduction = torch.sum
        #         # redun2_layer2.update_rule.reduction = torch.sum
        #         self.add_connection(redun1_layer2, source=f"Ae_{i-1}_r1", target=f"Ae_{i}")
        #         # self.add_connection(redun2_layer2, source=f"Ae_{i-1}_r2", target=f"Ae_{i}")
        #     input_layer = exc_layer
            
    # def __init__(
    #     self,
    #     n_inpt: int,
    #     n_neurons: int = 100,
    #     num_layers: int = 10,
    #     exc: float = 22.5,
    #     inh: float = 17.5,
    #     dt: float = 1.0,
    #     nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
    #     reduction: Optional[callable] = None,
    #     wmin: float = 0.0,
    #     wmax: float = 1.0,
    #     norm: float = 78.4,
    #     theta_plus: float = 0.05,
    #     tc_theta_decay: float = 1e7,
    #     inpt_shape: Optional[Iterable[int]] = None,
    #     inh_thresh: float = -40.0,
    #     exc_thresh: float = -52.0,
    #     if_inh = False,
    # ) -> None:
    #     super().__init__(dt=dt)

    #     self.n_inpt = n_inpt
    #     self.inpt_shape = inpt_shape
    #     self.n_neurons = n_neurons
    #     self.num_layers = num_layers
    #     self.exc = exc
    #     self.inh = inh
    #     self.dt = dt
        
    #     # Layers and connections lists
    #     self.input_layers = []
    #     self.exc_layers = []
    #     self.inh_layers = []
    #     self.input_exc_connections = []
    #     self.exc_inh_connections = []
    #     self.inh_exc_connections = []
    #     layer_names = []
        
    #     # Create input layer
    #     input_layer = Input(
    #         n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
    #     )
    #     # self.input_layers.append(input_layer)
    #     # layer_names.append("X")
    #     self.add_layer(input_layer, name="X")
        
        
    #     for i in range(self.num_layers):
    #         # Create excitatory layer
    #         # exc_layer = DiehlAndCookNodes(
    #         #     n=self.n_neurons,
    #         #     traces=True,
    #         #     rest=-65.0,
    #         #     reset=-60.0,
    #         #     thresh=exc_thresh,
    #         #     refrac=5,
    #         #     tc_decay=100.0,
    #         #     tc_trace=20.0,
    #         #     theta_plus=theta_plus,
    #         #     tc_theta_decay=tc_theta_decay,
    #         # )
    #         exc_layer = LIFNodes(
    #             n=self.n_neurons,
    #             traces=True,
    #             rest=-65.0,
    #             reset=-60.0,
    #             thresh=exc_thresh,#+(i/10*exc_thresh),
    #             refrac=5,
    #             tc_decay=100.0,
    #             tc_trace=20.0,
    #             theta_plus=theta_plus,
    #             tc_theta_decay=tc_theta_decay,
    #         )
    #         # self.exc_layers.append(exc_layer)
    #         # layer_names.append(f"Ae_{i}")
    #         self.add_layer(exc_layer, name=f"Ae_{i}")
    #         conn = None
    #         # Create connections
    #         if i == 0:  # If it's the first layer
    #             w_input_exc = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
    #             source_layer = "X"  # Source is the input layer
    #             conn = Connection(
    #                 source=self.layers[source_layer],  # Use the appropriate source layer
    #                 target=exc_layer,
    #                 w=w_input_exc,
    #                 update_rule=PostPre,
    #                 nu=nu,
    #                 reduction=reduction,
    #                 wmin=wmin,
    #                 wmax=wmax,
    #                 norm=norm,
    #                 )
    #             # self.input_exc_connections.append(conn)
    #         else:
    #             w_input_exc = 0.3 * torch.rand(self.n_neurons, self.n_neurons)
    #             source_layer = f"Ae_{i-1}"  # Source is the previous excitatory layer
    #             conn = Connection(
    #                 source=self.layers[source_layer],  # Use the appropriate source layer
    #                 target=exc_layer,
    #                 w=w_input_exc,
    #                 update_rule=PostPre,
    #                 nu=nu,
    #                 reduction=reduction,
    #                 wmin=wmin,
    #                 wmax=wmax,
    #                 norm=norm,
    #                 )
    #             # self.input_exc_connections.append(conn)
            
    #         if(i == num_layers-1 and if_inh):
    #             # Create inhibitory layer
    #             inh_layer = LIFNodes(
    #                 n=self.n_neurons,
    #                 traces=False,
    #                 rest=-60.0,
    #                 reset=-45.0,
    #                 thresh=inh_thresh,
    #                 tc_decay=10.0,
    #                 refrac=2,
    #                 tc_trace=20.0,
    #             )
    #             # self.inh_layers.append(inh_layer)
    #             # layer_names.append(f"Ai_{i}")
    #             self.add_layer(inh_layer, name=f"Ai_{i}")
            
    #             w_exc_inh = self.exc * torch.diag(torch.ones(self.n_neurons))
    #             exc_inh_conn = Connection(
    #                 source=exc_layer, target=inh_layer, w=w_exc_inh, wmin=0, wmax=self.exc
    #             )
    #             self.exc_inh_connections.append(exc_inh_conn)
            
    #             w_inh_exc = -self.inh * (
    #                 torch.ones(self.n_neurons, self.n_neurons)
    #                 - torch.diag(torch.ones(self.n_neurons))
    #             )
    #             inh_exc_conn = Connection(
    #                 source=inh_layer, target=exc_layer, w=w_inh_exc, wmin=-self.inh, wmax=0
    #             )
    #             self.inh_exc_connections.append(inh_exc_conn)
    #             self.add_connection(exc_inh_conn, source=f"Ae_{i}", target=f"Ai_{i}")
    #             self.add_connection(inh_exc_conn, source=f"Ai_{i}", target=f"Ae_{i}")
    #         if(i==0):
    #             self.add_connection(conn, source="X", target=f"Ae_{i}")
    #         else:
    #             self.add_connection(conn, source=f"Ae_{i-1}", target=f"Ae_{i}")
    #             # curr_l = i;
    #             # while((i-1)>0):
    #             #     i=i-1
    #             #     if(i+1 != curr_l):
    #             #         self.add_connection(input_exc_conn, source=f"Ae_{i-1}", target=f"Ae_{curr_l}")
                    
        
    #         # Set the next input layer to the current excitatory layer
    #         input_layer = exc_layer