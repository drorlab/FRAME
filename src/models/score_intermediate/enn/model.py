from functools import partial
import collections as col

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import e3nn

from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel


class E3NN_Conv_0to2(nn.Module):
    def __init__(self, in_filter, out_filter, radial_model,
                 act_fn, norm_fn):
        super().__init__()

        self.act = act_fn
        self.norm = norm_fn

        ### NOTE: Assume input order is always 1 and output order is always 2
        in_Rs = [(in_filter, 0), (in_filter, 1), (in_filter, 2)]
        out_Rs = [(out_filter, 0), (out_filter, 1), (out_filter, 2)]

        ### Self-interaction layer before the convolution layer; only mix
        # within channel
        self.in_lin0 = Linear([in_Rs[0]], [out_Rs[0]])

        ### Convolution layer
        # kernel: composed on a radial part that contains the learned
        # parameters and an angular part given by the spherical hamonics and
        # the Clebsch-Gordan coefficients
        selection_rule = partial(o3.selection_rule_in_out_sh, lmax=2)
        K = partial(
            Kernel, RadialModel=radial_model, selection_rule=selection_rule)

        self.conv10 = Convolution(K([out_Rs[0]], [out_Rs[0]]))
        self.conv11 = Convolution(K([out_Rs[0]], [out_Rs[1]]))
        self.conv12 = Convolution(K([out_Rs[0]], [out_Rs[2]]))

        ### Self-interaction layer after convolution layer; only mix within channel
        self.out_lin0 = Linear([out_Rs[0]], [out_Rs[0]])
        self.out_lin1 = Linear([out_Rs[1]], [out_Rs[1]])
        self.out_lin2 = Linear([out_Rs[2]], [out_Rs[2]])

        ### Non-linearities
        self.out_nonlin0 = Nonlinearity([out_Rs[0]], act=self.act)
        self.out_nonlin1 = Nonlinearity([out_Rs[1]], act=self.act)
        self.out_nonlin2 = Nonlinearity([out_Rs[2]], act=self.act)

    def forward(self, x, edge_index, edge_attr):
        out = self.in_lin0(x[0])

        # edge_index and edge_attr encode the neighborhood information.
        out0 = self.conv10(out, edge_index, edge_attr)
        out1 = self.conv11(out, edge_index, edge_attr)
        out2 = self.conv12(out, edge_index, edge_attr)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.out_lin0(out0)
        out1 = self.out_lin1(out1)
        out2 = self.out_lin2(out2)

        out0 = self.out_nonlin0(out0)
        out1 = self.out_nonlin1(out1)
        out2 = self.out_nonlin2(out2)
        return (out0, out1, out2)


class E3NN_Conv_2to2(nn.Module):
    def __init__(self, in_filter, out_filter, radial_model,
                 act_fn, norm_fn):
        super().__init__()

        self.act = act_fn
        self.norm = norm_fn

        ### NOTE: Assume input and output order are always 2
        in_Rs = [(in_filter, 0), (in_filter, 1), (in_filter, 2)]
        out_Rs = [(out_filter, 0), (out_filter, 1), (out_filter, 2)]

        ### Self-interaction layer before the convolution layer; only mix
        # within channel
        self.in_lin0 = Linear([in_Rs[0]], [out_Rs[0]])
        self.in_lin1 = Linear([in_Rs[1]], [out_Rs[1]])
        self.in_lin2 = Linear([in_Rs[2]], [out_Rs[2]])

        ### Convolution layer

        # Hack to ensure we are not mixing the input/output radial channels and
        # the filter channels to match the TFN implementation, which is more
        # efficient compared to that of e3nn (e3nn does some weird stuffs mixing
        # the input/output/filter channels which slows thing down tremendously).
        #
        # TODO(psuriana): we might be able to remove this hack with the latest
        # e3nn version using the "uvu" TensorProduct
        def filterfn_def(x, f):
            return x == f

        self.conv = nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=radial_model,
                                selection_rule=selection_rule)
                    self.conv[str((i, f, o))] = Convolution(K([out_Rs[i]], [out_Rs[o]]))

        ### Self-interaction layer after convolution layer; only mix within channel
        # To account for multiple output paths of conv (i.e. there are multiple
        # ways to get order 0, 1, 2, etc.).
        # Note: we don't need to do this on the first conv layer since
        # there is only scalar input (order 0).
        self.out_lin0 = Linear([(3 * out_filter, 0)], [out_Rs[0]])
        self.out_lin1 = Linear([(6 * out_filter, 1)], [out_Rs[1]])
        self.out_lin2 = Linear([(6 * out_filter, 2)], [out_Rs[2]])

        ### Non-linearities
        self.out_nonlin0 = Nonlinearity([out_Rs[0]], act=self.act)
        self.out_nonlin1 = Nonlinearity([out_Rs[1]], act=self.act)
        self.out_nonlin2 = Nonlinearity([out_Rs[2]], act=self.act)

    def forward(self, x, edge_index, edge_attr):
        out0 = self.in_lin0(x[0])
        out1 = self.in_lin1(x[1])
        out2 = self.in_lin2(x[2])

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        # Perform the convolution per order and combine the df per order.
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    curr = self.conv[str((i, f, o))](ins[i], edge_index, edge_attr)
                    tmp[o].append(curr)
        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.out_lin0(out0)
        out1 = self.out_lin1(out1)
        out2 = self.out_lin2(out2)

        out0 = self.out_nonlin0(out0)
        out1 = self.out_nonlin1(out1)
        out2 = self.out_nonlin2(out2)
        return (out0, out1, out2)


class E3NN_Model(nn.Module):

    def __init__(self, in_dim, rbf_high, rbf_count, num_nearest_neighbors,
                 e3nn_filters, fc_filters, out_dim=1, **kwargs):
        super().__init__()

        self.dataset = kwargs['dataset']
        ## NOTE: Assume that ligand flag (if applicable) is added at the end of
        # the one-hot-encoding input features followed by the fragment flag (if applicable)
        self.use_channel = None
        if 'agg_focus' in kwargs:
            if kwargs['agg_focus']:
                self.use_channel = -1
        '''
            assert not kwargs['no_ligand_flag']
            self.use_channel = -1 if not kwargs['flag_fragment'] else -2
        elif self.dataset == 'fragment_avg_fragment':
            assert kwargs['flag_fragment']
            self.use_channel = -1
        '''

        self.norm = Norm()
        self.ssp = ShiftedSoftplus()
        # Radial model:  R+ -> R^d
        # GaussianRadialModel is another NN: L is the number of layers, and
        # h is the size of the hidden layer
        relu = nn.ReLU()
        self.radial_model = partial(
            GaussianRadialModel,
            max_radius=rbf_high,
            number_of_basis=rbf_count,
            h=12, L=1, act=relu)

        ### Define network
        in_filter = in_dim

        # E3NN convolution layers
        e3nn_layers = []
        for i, out_filter in enumerate(e3nn_filters):
            if i == 0:
                e3nn_layers.append(E3NN_Conv_0to2(
                    in_filter, out_filter, self.radial_model, self.ssp, self.norm))
            else:
                e3nn_layers.append(E3NN_Conv_2to2(
                    in_filter, out_filter, self.radial_model, self.ssp, self.norm))
            in_filter = out_filter
        self.e3nn_net = nn.Sequential(*e3nn_layers)

        # FC layers
        fc_layers = []
        for out_filter in fc_filters:
            fc_layers.extend([
                nn.Linear(in_filter, out_filter, bias=True),
                nn.ELU(),
                ])
            in_filter = out_filter
        # Final FC layer
        last_fc = nn.Linear(in_filter, out_dim, bias=True)
        fc_layers.append(last_fc)
        self.dense_net = nn.Sequential(*fc_layers)

    def forward(self, data):
        ### Convolution layers
        out = [data.x]
        for layer in self.e3nn_net:
            out = layer(out, data.edge_index, data.edge_attr)

        # The last 2 channels of the input feature should be the ligand
        # flag followed by the fragment flags
        ### Per-channel mean
        if self.use_channel != None:
            selected = torch.nonzero(data.x[:,self.use_channel])
            batch = torch.squeeze(data.batch[selected])
            '''unique = torch.unique(batch)
            if data.label.shape != unique.shape:
                import pdb; pdb.set_trace()'''
            out = scatter_mean(torch.squeeze(out[0][selected]), batch, dim=0)
        else:
            out = scatter_mean(out[0], data.batch, dim=0)

        out = self.dense_net(out)
        out = torch.squeeze(out, axis=1)
        return out


class E3NN_Model_Per_Atom(E3NN_Model):

    def __init__(self, in_dim, rbf_high, rbf_count, num_nearest_neighbors,
                 e3nn_filters, fc_filters, **kwargs):
        super().__init__(in_dim, rbf_high, rbf_count, num_nearest_neighbors,
                         e3nn_filters, fc_filters, **kwargs)

    def forward(self, data):
        ### Convolution layers
        print("in per atom model")
        out = [data.x]
        for layer in self.e3nn_net:
            out = layer(out, data.edge_index, data.edge_attr)

        # Keep only some selected points
        out = torch.squeeze(out[0][data.select_atoms_index])
        out = self.dense_net(out)
        return out

class E3NN_Model_Frag_Type(E3NN_Model):
    def __init__(self, in_dim, rbf_high, rbf_count, num_nearest_neighbors,
                 e3nn_filters, fc_filters, **kwargs):
        self.out_dim = 60
        super().__init__(in_dim, rbf_high, rbf_count, num_nearest_neighbors,
                         e3nn_filters, fc_filters, out_dim=self.out_dim, **kwargs)

        if kwargs['agg_focus']:
            self.use_channel = -1



class ShiftedSoftplus:
    def __init__(self):
        self.shift = torch.nn.functional.softplus(torch.zeros(())).item()

    def __call__(self, x):
        return torch.nn.functional.softplus(x).sub(self.shift)


######################## With pre-trained E3NN model ########################

class E3NN_Transfer_Model(nn.Module):

    def __init__(self, in_filter, e3nn_net, fc_filters, freeze_pretrained, **kwargs):
        super().__init__()

        # Pre-trained e3nn pdbbind model
        self.e3nn_net = e3nn_net

        # Freeze the e3nn layers
        if freeze_pretrained:
            for param in self.e3nn_net.parameters():
                param.requires_grad = False

        # FC layers
        fc_layers = []
        for out_filter in fc_filters:
            fc_layers.extend([
                nn.Linear(in_filter, out_filter, bias=True),
                nn.ELU(),
                ])
            in_filter = out_filter
        # Final FC layer
        last_fc = nn.Linear(in_filter, 1, bias=True)
        fc_layers.append(last_fc)
        self.dense_net = nn.Sequential(*fc_layers)

    def forward(self, d):
        ### Convolution layers
        out = [d.x]
        for layer in self.e3nn_net:
            out = layer(out, d.edge_index, d.edge_attr)

        ### Per-channel mean
        out = scatter_mean(out[0], d.batch, dim=0)

        out = self.dense_net(out)
        out = torch.squeeze(out, axis=1)
        return out

