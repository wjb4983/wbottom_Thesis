import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=1)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=100)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.15)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
n_updates = args.n_updates
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu

update_steps = int(n_train / batch_size / n_updates)
update_interval = update_steps * batch_size
print(update_interval)

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from bindsnet.utils import im2col_indices
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv1dConnection,
    Conv2dConnection,
    Conv3dConnection,
    LocalConnection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
)
from bindsnet.learning.learning import LearningRule
class CustomPostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        elif isinstance(connection, Custom_Connection):
            self.update = self._custom_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        target_x = self.target.x.reshape(batch_size, out_channels * height_out, 1)
        target_x = target_x * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_x = target_x * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )
        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_x = target_x * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_x = (
            self.source.x.unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any():
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        super().update()
        
    def _custom_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]

            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any():
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(self.source.x, _pair(padding))
        source_x = source_x.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            # print(self.nu[0].shape, self.connection.w.size())
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv3dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(
            self.source.x,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_x = (
            source_x.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()
        # print(target_x.shape, source_s.shape, self.connection.w.shape)

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import Module, Parameter

from bindsnet.network.topology import AbstractConnection
from bindsnet.network.nodes import CSRMNodes, Nodes
class Custom_Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        # import random
        # s_float = s.float()
        # # print(s.shape)  # Print tensor shape for debugging
        # mask = (s_float == 1) & (torch.rand_like(s_float) < 0.0)  # Create a mask for spike modification
        # s[mask] = 0  # Set spikes to 0 where the mask is True
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()



from typing import Iterable, Optional, Union
from bindsnet.network.nodes import DiehlAndCookNodes
class Custom_DiehlAndCookNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        batch_size: Union[float, torch.Tensor] = None,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )
        self.batch_size = batch_size
        if isinstance(thresh, torch.Tensor) and thresh.ndim == 2:
            self.register_buffer("thresh", thresh)  # Spike threshold voltage.
        else:
            # Expand the single threshold value to match (batch_size, num_neurons) shape
            self.register_buffer("thresh", torch.full((self.batch_size, self.n), thresh))

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        # self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.
        self.lbound = lbound  # Lower bound of voltage.
        self.one_spike = one_spike  # One spike per timestep.
        self.original_thresh = thresh

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(
                    self.s.float().view(self.batch_size, -1)[_any], 1
                )
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
        
#Diehl and cook nodes
class Custom_adLIFNeurons(Custom_DiehlAndCookNodes):
    def __init__(self, *args, spike_limit=20, device = 'cuda', **kwargs):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        self.spike_limit = spike_limit
        

    def forward(self, *args, **kwargs):
        # Increment spike count for each batch element
        self.spike_counts += self.s.sum(dim=1, keepdim=True)
        # print(self.s.sum(dim=1, keepdim=True), self.spike_counts)
        # print(torch.unique(self.thresh))
        # Check if spike limit reached for any batch element
        exceeding_spikes = self.spike_counts >= self.spike_limit
        if exceeding_spikes.any():
            exceeding_indices = torch.nonzero(self.spike_counts >= self.spike_limit, as_tuple=False)
            batch_indices = exceeding_indices[:, 0]  # Get the batch indices
            unique_batches = torch.unique(batch_indices)  # Get unique batch indices
            # print(unique_batches)
            # Set threshold to inf for exceeding batch elements
            for batch_idx in unique_batches:
                self.thresh[batch_idx] = float('inf')
        
        super().forward(*args, **kwargs)
    def reset_state_variables(self) -> None:
        # Reset state variables including spike count
        super().reset_state_variables()
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        # print(self.thresh)
        self.thresh.fill_(self.original_thresh)
        # print(self.original_thresh)
        # print(self.thresh)
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
class DiehlAndCook2015_CustomConnection(Network):
    


    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = None,#(1e-4, 1e-4),
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
        exc_layer = Custom_adLIFNeurons(
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
            batch_size=batch_size
        )
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

        # Connections
        w = 0.8 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            # norm=norm/6,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")





# Build network.
network = DiehlAndCook2015_CustomConnection(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

w1 = network.connections[("X", "Ae")].w
print(w1.mean().item())
# Directs network to GPU
if gpu:
    network.to("cuda")
    
from SortedMNIST import SortedMNIST

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../../data/MNIST",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training...")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("\nProgress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    pbar_training = tqdm(total=n_train)
    for step, batch in enumerate(train_dataloader):
        if step * batch_size > n_train:
            break

        # Assign labels to excitatory neurons.
        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []
            # if(accuracy["all"][-1]>50.0):
            #     network.connections[('X', 'Ae')].update_rule.nu = (1e-4,1e-4)#network.connections[('X', 'Ae')].update_rule.nu/100

        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Remember labels.
        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time)  

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][:, 0].view(28, 28)
            inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
            lable = batch["label"][0]
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            }
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy, x_scale=update_steps * batch_size, ax=perf_ax
            )
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            # plt.pause(1e-8)
        w1 = network.connections[("X", "Ae")].w
        print(w1.mean().item())
        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)
    pbar_training.close()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTraining complete.\n")

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")

for step, batch in enumerate(test_dataloader):
    if step * batch_size > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(
        torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
    )
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTesting complete.\n")