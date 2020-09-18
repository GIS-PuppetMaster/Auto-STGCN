from utils.layer_utils import *


class Model(HybridBlock):
    def __init__(self, action_trajectory, config, ctx, adj_SIPMs):
        super(Model, self).__init__()
        self.ctx = ctx
        # only contains stage2 and stage3 action
        self.action_trajectory = action_trajectory

        self.batch_size = config['batch_size']
        self.phi = config['phi']
        self.num_of_vertices = config['num_of_vertices']
        self.adj_filename = config['adj_filename']
        self.id_filename = config['id_filename']
        self.module_type = "individual"
        self.n = config['n']
        self.order_of_cheb = 3
        self.config = config
        # infer the number of end-block_unit
        tmp_output_block_list = [None for _ in range(len(action_trajectory))]
        for i, action in enumerate(action_trajectory):
            if i == 0:
                continue
            input_block = action[3]
            tmp_output_block_list[input_block] = i
        self.num_of_end_block = 0
        for i in range(len(tmp_output_block_list)):
            if tmp_output_block_list[i] is None:
                self.num_of_end_block += 1
        # layer_list[0] represents the output of stage 2
        # contains hybrid block or function reference
        # init every kind of adj_matrix that needed by all kinds of blocks
        with self.name_scope():
            self.layer_list = [None]
            self.adj_SIPM1 = adj_SIPMs[0]
            self.adj_SIPM4 = adj_SIPMs[1]
            self.laplace_weights = None
            # visit action_trajectory and build all block
            for i, action in enumerate(self.action_trajectory):
                block_unit = []
                # i=0 => state -1
                # i=1 => state 0
                # i=2...n+1 => state 1...n
                # i=n+2 => state n+1
                if i == 0:
                    # current_state = -1, take action to decide the state 0
                    # IS1 will be applied in forward process, IS2 is None
                    if action[0] == 1:
                        temp = IS1()
                        self.layer_list[0] = temp
                        self.register_child(temp)
                    elif action[0]==2:
                        self.layer_list[0] = None
                    else:
                        raise Exception(f'IS error, got:{action[0]}, with action_trajectory:{self.action_trajectory}')

                    self.filter_size = action[2]

                    if action[1] == 1:
                        self.output_structure = OS2(self.filter_size)
                    elif action[1]==2:
                        self.output_structure = OS3(self.num_of_vertices)
                    else:
                        raise Exception(f'OS error, got:{action[1]}, with action_trajectory:{self.action_trajectory}')

                    if action[3] == 1:
                        self.MOBF = MOBF1(self.num_of_end_block, self.num_of_vertices)
                    elif action[3]==2:
                        self.MOBF = MOBFEmbedding(self.num_of_end_block, self.num_of_vertices, False)
                    else:
                        raise Exception(f'MOBF error, got:{action[3]}, with action_trajectory:{self.action_trajectory}')
                elif i <= self.n + 1:
                    # current_state = 0/1...n-1, take action to decide the state 1...n
                    if action[0] == 1:
                        # SIPM1, will be applied when building the block which needs adj matrix
                        # use Pearson Adj matrix for all layer
                        # hold position
                        block_unit.append("SIPM1")
                    elif action[0] == 2:
                        # SIPM2, Spatial attention
                        block_unit.append(SIPM2())
                        self.register_child(block_unit[-1])
                    elif action[0] == 3:
                        # SIPM3, will be applied when building the block which needs adj matrix
                        # use mask weight Adj matrix for all layer
                        # hold position
                        block_unit.append(SIPM3(num_of_vertices=self.num_of_vertices))
                        self.register_child(block_unit[-1])
                    elif action[0] == 4:
                        # SIPM4, None
                        # use original adj matrix
                        # hold position
                        block_unit.append("SIPM4")

                    if action[1] == 1:
                        # TIPM1
                        block_unit.append(TIPM1())
                        self.register_child(block_unit[-1])
                    elif action[1] == 2:
                        # TIPM2
                        block_unit.append(TIPM2(self.num_of_vertices))
                        self.register_child(block_unit[-1])
                    elif action[1] == 3:
                        # TIPM3, None
                        # hold position
                        block_unit.append(None)

                    if action[2] == 1:
                        # FES1
                        block_unit.append(
                            FES1(order_of_cheb=self.order_of_cheb, Kt=self.filter_size, channels=[64, 16, 64],
                                 num_of_vertices=self.num_of_vertices,
                                 keep_prob=1.0,
                                 batch_size=self.batch_size,
                                 activation='GLU'))
                        self.register_child(block_unit[-1])
                    elif action[2] == 2:
                        block_unit.append(FES2())
                        self.register_child(block_unit[-1])
                    elif action[2] == 3:
                        block_unit.append(FES3(3))
                        self.register_child(block_unit[-1])
                    elif action[2] == 4:
                        block_unit.append(FES4(self.num_of_vertices, [64, 64, 64], self.module_type, 'GLU'))
                        self.register_child(block_unit[-1])

                    assert len(block_unit) == 3
                    self.layer_list.append(block_unit)

    def block_unit(self, data, block_unit_layers):
        # laplace weights not used default
        self.laplace_weights = None
        for i, layer in enumerate(block_unit_layers):
            if layer is not None:
                if i == 0:
                    if isinstance(layer, SIPM2):
                        # SIPM2
                        self.laplace_weights = layer(data)
                        self.adj_choosed = self.adj_SIPM4
                    elif isinstance(layer, SIPM3):
                        self.adj_choosed = layer(self.adj_SIPM4, 1)
                    else:
                        if layer == "SIPM1":
                            self.adj_choosed = self.adj_SIPM1
                        elif layer == "SIPM4":
                            self.adj_choosed = self.adj_SIPM4
                elif i == 1:
                    # TIPM1-2
                    data = layer(data)
                elif i == 2:
                    # FES1-4
                    if isinstance(layer, FES1):
                        data = layer(data, nd.array(
                            cheb_poly_approx(scaled_Laplacian(self.adj_choosed), self.order_of_cheb)),
                                     self.laplace_weights)
                    elif isinstance(layer, FES2):
                        data = layer(data, self.adj_choosed, self.laplace_weights)
                    elif isinstance(layer, FES3):
                        L_tilde = scaled_Laplacian(self.adj_choosed)
                        cheb_polynomials = [nd.array(i)
                                            for i in cheb_polynomial(L_tilde, 3)]
                        data = layer(data, cheb_polynomials, self.laplace_weights)
                    elif isinstance(layer, FES4):
                        data = layer(data, self.adj_choosed, self.laplace_weights)
        return data

    def hybrid_forward(self, F, x, *args, **kwargs):
        self.adj_SIPM1 = nd.array(self.adj_SIPM1)
        self.adj_SIPM4 = nd.array(self.adj_SIPM4)
        # output_data_list keep the output of all block_units in order
        # output_date_list[i] is the output ndarray of the i-th block_unit in action_trajectory
        global x_tmp
        output_data_list = []
        # end_data_list keep the index of the output in output_data_list
        # which will finally be the input of the MOBFEmbedding layer either use MOBF1 or MOBF2
        end_data_list = []
        for i, action in enumerate(self.action_trajectory):
            if i == 0:
                # IS
                if self.layer_list[0] is not None:
                    # IS1
                    x = self.layer_list[0](x)
                output_data_list.append(x)
                end_data_list.append(0)
                assert len(output_data_list) - 1 == i
            else:
                # find the i-th block unit from self.layer_list
                block_unit = self.layer_list[i]
                # dqn guarantee action[3]<i, which must be in the end_data_list
                input_data = output_data_list[action[3]]
                x = self.block_unit(input_data, block_unit)
                output_data_list.append(x)
                if action[3] in end_data_list:
                    end_data_list.remove(action[3])
                end_data_list.append(i)
                assert len(output_data_list) - 1 == i
        # MOBF1 and MOBF2 both need to adjust feature axis to the same size
        x = self.MOBF(end_data_list, output_data_list)
        # MOBF2
        if isinstance(self.MOBF, MOBFEmbedding):
            for i in range(len(x)):
                if i == 0:
                    x_tmp = x[i]
                else:
                    x_tmp = nd.concat(x_tmp, x[i], dim=1)
            x = x_tmp
        x = self.output_structure(x)
        return x
