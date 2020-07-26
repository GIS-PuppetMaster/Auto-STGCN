import mxnet as mx
from mxnet.gluon.block import HybridBlock
import mxnet.ndarray as nd
from layer_utils import *


class Model(HybridBlock):
    def __init__(self, action_trajectory, config, ctx, adj_SIPMs):
        super(Model, self).__init__()
        self.ctx = ctx
        self.action_trajectory = action_trajectory

        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']
        self.num_of_vertices = config['num_of_vertices']
        self.adj_filename = config['adj_filename']
        self.id_filename = config['id_filename']
        self.module_type = config['module_type']
        self.n = config['n']
        self.order_of_cheb = 3
        self.config = config
        # layer_list[0] represents the output of stage 2
        # contains hybrid block or function reference
        self.layer_list = [None]
        # init every kind of adj_matrix that needed by all kinds of blocks
        with self.name_scope():
            self.adj_SIPM1 = adj_SIPMs[0]
            self.adj_SIPM3 = adj_SIPMs[1]
            self.adj_SIPM4 = adj_SIPMs[2]
            # visit action_trajectory and build all block
            for i, action in enumerate(self.action_trajectory):
                if action[5] == 1:
                    # train hyper parameters
                    continue
                block_unit = []
                # i=0 => state -1
                # i=1 => state 0
                # i=2...n+1 => state 1...n
                # i=n+2 => state n+1
                if i == 0:
                    # current_state = -1, take action to decide the state 0
                    # IS1 will be applied in forward process, IS2 is None
                    if action[0] == 1:
                        self.layer_list[0] = IS1
                    else:
                        self.layer_list[0] = None

                    self.filter_size = action[2]

                    if action[1] == 1:
                        self.output_structure = OS1(self.num_of_vertices, self.batch_size)
                    elif action[1] == 2:
                        self.output_structure = OS2(self.filter_size)
                    else:
                        self.output_structure = OS3(self.num_of_vertices)

                    self.MOBF = action[3]

                elif i <= self.n + 1:
                    # current_state = 0, take action to decide the state 1...n
                    sipm = 4
                    if action[0] == 1:
                        # SIPM1, will be applied when building the block which needs adj matrix
                        # use Pearson Adj matrix for all layer
                        # hold position
                        block_unit.append("SIPM1")
                        sipm = 1
                    elif action[0] == 2:
                        # SIPM2, Spatial attention
                        block_unit.append(SIPM2())
                        sipm = 2
                    elif action[0] == 3:
                        # SIPM3, will be applied when building the block which needs adj matrix
                        # use mask weight Adj matrix for all layer
                        # hold position
                        block_unit.append("SIPM3")
                        sipm = 3
                    elif action[0] == 4:
                        # SIPM4, None
                        # use original adj matrix
                        # hold position
                        block_unit.append("SIPM4")
                        sipm = 4

                    if action[1] == 1:
                        # TIPM1
                        block_unit.append(TIPM1())
                    elif action[1] == 2:
                        # TIPM2
                        block_unit.append(TIPM2(self.num_of_vertices))
                    elif action[1] == 3:
                        # TIPM3, None
                        # hold position
                        block_unit.append(None)

                    # set adj
                    if sipm == 1:
                        adj_temp = self.adj_SIPM1
                    elif sipm == 2:
                        # process spatial information with SIPM2 block, use raw adj matrix
                        adj_temp = self.adj_SIPM4
                    elif sipm == 3:
                        adj_temp = self.adj_SIPM3
                    elif sipm == 4:
                        adj_temp = self.adj_SIPM4
                    if action[2] == 1:
                        # FES1
                        block_unit.append(
                            FES1(order_of_cheb=self.order_of_cheb, kt=self.filter_size, channels=[64, 16, 64],
                                 num_of_vertices=self.num_of_vertices,
                                 keep_prob=1.0,
                                 cheb_polys=nd.array(cheb_poly_approx(scaled_Laplacian(adj_temp), self.order_of_cheb)),
                                 activation='GLU'))
                    elif action[2] == 2:
                        block_unit.append(FES2(adj_temp))
                    elif action[2] == 3:
                        block_unit.append(FES3(12, get_backbones(self.config, adj_temp, self.ctx)))
                    elif action[2] == 4:
                        block_unit.append(FES4(adj_temp, self.num_of_vertices, [64, 16, 64], self.module_type, 'GLU'))

                    assert len(block_unit) == 3
                    self.layer_list.append(block_unit)

    def build_block_unit(self, data, block_unit_layers):
        for i, layer in enumerate(block_unit_layers):
            if layer is not None:
                if i == 0:
                    if not isinstance(layer, str):
                        # SIPM2
                        data = layer(data)
                elif i == 1:
                    # TIPM1-2
                    data = layer(data)
                elif i == 2:
                    # FES1-4
                    data = layer(data)
        return data

    def hybrid_forward(self, F, x, *args, **kwargs):
        end_data_list = []
        for i, action in enumerate(self.action_trajectory):
            if i == 0:
                if self.layer_list[0] is not None:
                    x = self.layer_list[0](x)
                end_data_list.append(x)
            else:
                for j, action in enumerate(self.action_trajectory):
                    if action[5] == 1:
                        # train hyper parameters
                        continue
                    block_to_connect = action[4]
                    end_data_list.remove(x)
                    x = self.build_block_unit(x, self.layer_list[block_to_connect])
                    end_data_list.append(x)
        if self.MOBF == 1:
            x = MOBF1Fuse(MOBF1(end_data_list, self.num_of_vertices))
        else:
            for i in range(len(end_data_list)):
                if i == 0:
                    x = end_data_list[i]
                else:
                    x = nd.concat(x, end_data_list[i], dim=1)
        x = self.output_structure(x)
        return x
