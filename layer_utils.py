from mxnet.gluon import HybridBlock, Block
from mxnet.gluon.nn import Activation, HybridSequential
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.rnn import HybridRecurrentCell
from scipy.stats import pearsonr
from mxnet.gluon import nn
import mxnet as mx
from scipy.sparse.linalg import eigs


def cheb_poly_approx(L, order_of_cheb):
    '''
    Chebyshev polynomials approximation

    Parameters
    ----------
    L: np.ndarray, scaled graph Laplacian,
       shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, order_of_cheb * num_of_vertices)

    '''
    if order_of_cheb == 1:
        return nd.diag(nd.ones_like(L[0, :]))

    cheb_polys = [nd.diag(nd.ones_like(L[0, :])), L]

    for i in range(2, order_of_cheb):
        cheb_polys.append(2 * L * cheb_polys[i - 1] - cheb_polys[i - 2])

    return nd.concat(*cheb_polys, dim=-1)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]
    D = nd.diag(nd.sum(W, axis=1))
    L = D - W

    lambda_max = nd.array(eigs(L.asnumpy(), k=1, which='LR')[0].real)

    return (2 * L) / lambda_max - nd.diag(nd.ones_like(W[0, :]))


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    cheb_polynomials = [nd.diag(nd.ones_like(L_tilde[0, :])), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class IS1(HybridBlock):
    def __init__(self):
        super(IS1, self).__init__()
        with self.name_scope():
            self.dense = mx.gluon.nn.Dense(units=64, activation='relu', flatten=False,
                                           weight_initializer=mx.initializer.Xavier())

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.dense(x)


def SIPM1(filepath=None, **kwargs):
    if filepath is None:
        time_series_matrix = kwargs['time_series_matrix']
        num_of_vertices = kwargs['num_of_vertices']
        epsilon = kwargs['epsilon']
        adj_matrix = np.zeros(shape=(int(num_of_vertices), int(num_of_vertices)))
        # construct new adj_matrix with pearson in CIKM paper
        for i in range(time_series_matrix.shape[1]):
            for j in range(time_series_matrix.shape[1]):
                if pearsonr(time_series_matrix[:, i], time_series_matrix[:, j])[0] > epsilon:
                    adj_matrix[i, j] = 1
        return adj_matrix
    else:
        return np.load(filepath)['arr_0']


class SIPM2(HybridBlock):
    '''
    compute spatial attention scores
    '''

    def __init__(self, **kwargs):
        super(SIPM2, self).__init__(**kwargs)
        with self.name_scope():
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.W_3 = self.params.get('W_3', allow_deferred_init=True)
            self.b_s = self.params.get('b_s', allow_deferred_init=True)
            self.V_s = self.params.get('V_s', allow_deferred_init=True)

    def forward(self, x, *args, **kwargs):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)

        '''
        x = x.transpose((0, 2, 3, 1))
        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer the shape of params
        self.W_1.shape = (num_of_timesteps,)
        self.W_2.shape = (num_of_features, num_of_timesteps)
        self.W_3.shape = (num_of_features,)
        self.b_s.shape = (1, num_of_vertices, num_of_vertices)
        self.V_s.shape = (num_of_vertices, num_of_vertices)
        for param in [self.W_1, self.W_2, self.W_3, self.b_s, self.V_s]:
            param._finish_deferred_init()

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = nd.dot(nd.dot(x, self.W_1.data()), self.W_2.data())

        # shape of rhs is (batch_size, T, V)
        rhs = nd.dot(self.W_3.data(), x.transpose((2, 0, 3, 1)))

        # shape of product is (batch_size, V, V)
        product = nd.batch_dot(lhs, rhs)

        S = nd.dot(self.V_s.data(),
                   nd.sigmoid(product + self.b_s.data())
                   .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normalization
        S = S - nd.max(S, axis=1, keepdims=True)
        exp = nd.exp(S)
        S_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return S_normalized


def construct_adj(A, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: numpy.array, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    diag_matrix = nd.diag(nd.ones_like(A)[0, :])
    zero_matrix = nd.zeros_like(A)
    line1 = nd.concat(*[A, diag_matrix, zero_matrix], dim=1)
    line2 = nd.concat(*[diag_matrix, A, diag_matrix], dim=1)
    line3 = nd.concat(*[zero_matrix, diag_matrix, A], dim=1)
    adj = nd.concat(*[line1, line2, line3], dim=0)
    return adj


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


class SIPM3(HybridBlock):
    def __init__(self, num_of_vertices):
        super(SIPM3, self).__init__()
        self.num_of_vertices = num_of_vertices
        with self.name_scope():
            self.mask = self.params.get("mask", allow_deferred_init=True)

    def forward(self, x, *args):
        adj = x
        step = args[0]
        if not isinstance(adj, nd.NDArray):
            adj = nd.array(adj)
        # self.x_shape = adj.shape
        # step = self.x_shape[-1] // adj.shape[-1]
        self.mask.shape = x.shape
        self.mask.init = mx.init.Constant(value=(adj != 0))
        self.mask._finish_deferred_init()
        return self.mask.data() * adj


def SIPM4(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    return get_adjacency_matrix(distance_df_filename, num_of_vertices, type_, id_filename)


class TIPM1(HybridBlock):
    '''
    compute temporal attention scores
    '''

    def __init__(self, **kwargs):
        super(TIPM1, self).__init__(**kwargs)
        with self.name_scope():
            self.U_1 = self.params.get('U_1', allow_deferred_init=True)
            self.U_2 = self.params.get('U_2', allow_deferred_init=True)
            self.U_3 = self.params.get('U_3', allow_deferred_init=True)
            self.b_e = self.params.get('b_e', allow_deferred_init=True)
            self.V_e = self.params.get('V_e', allow_deferred_init=True)

    def forward(self, x, *args, **kwargs):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        x = x.transpose((0, 2, 3, 1))

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer shape
        self.U_1.shape = (num_of_vertices,)
        self.U_2.shape = (num_of_features, num_of_vertices)
        self.U_3.shape = (num_of_features,)
        self.b_e.shape = (1, num_of_timesteps, num_of_timesteps)
        self.V_e.shape = (num_of_timesteps, num_of_timesteps)
        for param in [self.U_1, self.U_2, self.U_3, self.b_e, self.V_e]:
            param._finish_deferred_init()

        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = nd.dot(nd.dot(x.transpose((0, 3, 2, 1)), self.U_1.data()),
                     self.U_2.data())

        # shape is (N, V, T)
        rhs = nd.dot(self.U_3.data(), x.transpose((2, 0, 1, 3)))

        product = nd.batch_dot(lhs, rhs)

        E = nd.dot(self.V_e.data(),
                   nd.sigmoid(product + self.b_e.data())
                   .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normailzation
        E = E - nd.max(E, axis=1, keepdims=True)
        exp = nd.exp(E)
        # shape is (batch_size, T_{r-1}, T_{r-1})
        E_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        x_TAt = nd.batch_dot(x.reshape(batch_size, -1, num_of_timesteps),
                             E_normalized) \
            .reshape(batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)
        x_TAt = x_TAt.transpose((0, 3, 1, 2))
        # shape is (batch_size, time_steps, vertices, feature)
        return x_TAt


class TIPM2(HybridBlock):
    '''
        Parameters
        ----------
        data: NDarray, shape is (B, T, N, C)

        input_length: int, length of time series, T

        num_of_vertices: int, N

        embedding_size: int, C

        temporal, spatial: bool, whether equip this type of embeddings

        init: mx.initializer.Initializer

        prefix_: str

        Returns
        ----------
        data: output shape is (B, T, N, C)
    '''

    def __init__(self, num_of_vertices, init=mx.init.Xavier(magnitude=0.0003), prefix_=""):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.init = init
        self._prefix = prefix_
        # shape is (1, T, 1, C)
        self.temporal_emb = self.params.get("{}_t_emb".format(self._prefix), allow_deferred_init=True)
        # shape is (1, 1, N, C)
        self.spatial_emb = self.params.get("{}_v_emb".format(self._prefix), allow_deferred_init=True)

    def forward(self, x, *args, **kwargs):
        input_length = x.shape[1]
        embedding_size = x.shape[-1]
        self.temporal_emb.shape = (1, input_length, 1, embedding_size)
        self.temporal_emb._finish_deferred_init()
        self.spatial_emb.shape = (1, 1, self.num_of_vertices, embedding_size)
        self.spatial_emb._finish_deferred_init()
        x = nd.broadcast_add(x, self.temporal_emb.data())
        x = nd.broadcast_add(x, self.spatial_emb.data())
        return x


class Gconv(HybridBlock):
    def __init__(self, order_of_cheb, c_in, c_out, num_of_vertices, batch_size, **kwargs):
        super(Gconv, self).__init__(**kwargs)
        self.order_of_cheb = order_of_cheb
        self.c_in = c_in
        self.c_out = c_out
        self.batch_size = batch_size
        self.num_of_vertices = num_of_vertices
        with self.name_scope():
            self.theta = nn.Dense(c_out, activation=None, flatten=False)

    def hybrid_forward(self, F, x, cheb_polys, spatial_At):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size * time_step, num_of_vertices, c_in)

        cheb_polys: nd.array,
                shape is (num_of_vertices, order_of_cheb * num_of_vertices)

        Returns
        ----------
        shape is (batch_size * time_step, num_of_vertices, c_out)
        '''
        if spatial_At is not None:
            spatial_At = nd.repeat(spatial_At, repeats=self.order_of_cheb, axis=-1)
            cheb_polys = cheb_polys * spatial_At
        # (batch_size, time_step * c_in, num_of_vertices)
        x_tmp = x.reshape((self.batch_size, -1, self.num_of_vertices, self.c_in)).transpose((0, 1, 3, 2)).reshape(
            (self.batch_size, -1, self.num_of_vertices))
        # x_tmp = x.transpose((0, 2, 1)).reshape((-1, self.num_of_vertices))
        if spatial_At is not None:
            # (batch_size, c_in, order_of_cheb, num_of_vertices)
            x_mul = F.batch_dot(x_tmp, cheb_polys).reshape((-1, self.c_in, self.order_of_cheb, self.num_of_vertices))
        else:
            x_mul = F.dot(x_tmp, cheb_polys).reshape((-1, self.c_in, self.order_of_cheb, self.num_of_vertices))

        # batch_size, num_of_vertices, c_in * order_of_cheb
        x_ker = x_mul.transpose((0, 3, 1, 2)) \
            .reshape((-1, self.num_of_vertices,
                      self.c_in * self.order_of_cheb))

        x_gconv = self.theta(x_ker)
        return x_gconv


class Temporal_conv_layer(HybridBlock):
    def __init__(self, Kt, c_in, c_out, activation='relu', **kwargs):
        super(Temporal_conv_layer, self).__init__(**kwargs)
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.activation = activation
        with self.name_scope():
            if c_in > c_out:
                self.res_conv = nn.Conv2D(c_out, kernel_size=(1, 1),
                                          activation=None, use_bias=False)
            if activation == 'GLU':
                self.conv = nn.Conv2D(2 * c_out, (Kt, 1), activation=None)
            elif activation == 'relu':
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=None)
            else:
                self.conv = nn.Conv2D(c_out, (Kt, 1), activation=activation)

    def hybrid_forward(self, F, x):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step - Kt + 1, num_of_vertices)

        '''

        if self.c_in == self.c_out:
            x_input = x
        elif self.c_in > self.c_out:
            x_input = self.res_conv(x)
        else:
            padding = F.broadcast_axis(
                F.slice(
                    F.zeros_like(x),
                    begin=(None, None, None, None),
                    end=(None, 1, None, None)
                ), axis=1, size=self.c_out - self.c_in)
            x_input = F.concat(x, padding, dim=1)

        x_input = F.slice(x_input,
                          begin=(None, None, self.Kt - 1, None),
                          end=(None, None, None, None))

        x_conv = self.conv(x)
        if self.activation == 'GLU':
            x_conv = self.conv(x)
            x_conv1 = F.slice(x_conv,
                              begin=(None, None, None, None),
                              end=(None, self.c_out, None, None))
            x_conv2 = F.slice(x_conv,
                              begin=(None, self.c_out, None, None),
                              end=(None, None, None, None))
            return (x_conv1 + x_input) * F.sigmoid(x_conv2)
        if self.activation == 'relu':
            return F.relu(x_conv + x_input)
        return x_conv


class Spatio_conv_layer(HybridBlock):
    def __init__(self, order_of_cheb, c_in, c_out, num_of_vertices, batch_size,
                 **kwargs):
        super(Spatio_conv_layer, self).__init__(**kwargs)
        self.order_of_cheb = order_of_cheb
        self.c_in = c_in
        self.c_out = c_out
        self.num_of_vertices = num_of_vertices
        with self.name_scope():
            if c_in > c_out:
                self.res_conv = nn.Conv2D(c_out, kernel_size=(1, 1),
                                          activation=None, use_bias=False)
            self.gconv = Gconv(order_of_cheb, c_in, c_out, num_of_vertices, batch_size)

    def hybrid_forward(self, F, x, cheb_polys, spatial_At):
        '''
        Parameters
        ----------
        x: nd.array, shape is (batch_size, c_in, time_step, num_of_vertices)

        cheb_polys: nd.array,
                shape is (num_of_vertices, order_of_cheb * num_of_vertices)

        Returns
        ----------
        shape is (batch_size, c_out, time_step, num_of_vertices)
        '''
        if self.c_in == self.c_out:
            x_input = x
        elif self.c_in > self.c_out:
            x_input = self.res_conv(x)
        else:
            padding = F.broadcast_axis(F.zeros_like(x), axis=1,
                                       size=self.c_out - self.c_in)
            x_input = F.concat(x, padding, dim=1)

        x_tmp = x.transpose((0, 2, 3, 1)) \
            .reshape((-1, self.num_of_vertices, self.c_in))

        x_gconv = self.gconv(x_tmp, cheb_polys, spatial_At)
        self.T = x.shape[2]
        x_gc = x_gconv.reshape((-1, self.T, self.num_of_vertices, self.c_out)) \
            .transpose((0, 3, 1, 2))

        x_gc = F.slice(x_gc,
                       begin=(None, None, None, None),
                       end=(None, self.c_out, None, None))
        return F.relu(x_gc + x_input)


class FES1(HybridBlock):
    def __init__(self, order_of_cheb, Kt, channels, num_of_vertices, keep_prob, batch_size,
                 activation='GLU', **kwargs):
        super(FES1, self).__init__(**kwargs)
        self.c_si, self.c_t, self.c_oo = channels
        self.activation = activation
        self.num_of_vertices = num_of_vertices
        self.order_of_cheb = order_of_cheb
        self.Kt = Kt
        self.keep_prob = keep_prob
        self.tcl1 = Temporal_conv_layer(self.Kt, self.c_si, self.c_t, self.activation)
        self.scl = Spatio_conv_layer(self.order_of_cheb, self.c_t, self.c_t,
                                     self.num_of_vertices, batch_size)
        self.tcl2 = Temporal_conv_layer(self.Kt, self.c_t, self.c_oo)
        self.normal = nn.LayerNorm(axis=1)
        self.dropout = nn.Dropout(1 - keep_prob)

    def hybrid_forward(self, F, x, cheb_polys, spatial_At):
        '''
        Parameters
        ----------
        x: nd.array,
           shape is (batch_size, channels[0], time_step, num_of_vertices)

        Returns
        ----------
        shape is (batch_size, time_step - 2(Kt - 1), num_of_vertices,  channels[-1],)
        '''
        x = x.transpose((0, 3, 1, 2))
        x = self.tcl1(x)
        x = self.scl(x, cheb_polys, spatial_At)
        x = self.tcl2(x)
        x = self.normal(x)
        x = self.dropout(x)
        x = x.transpose((0, 2, 3, 1))
        return x


class FES2(HybridBlock):
    def __init__(self):
        super(FES2, self).__init__()
        with self.name_scope():
            self.gnn_layer = BasicGNN()

    def forward(self, x, *args):
        # input:
        # nd.NDArray, x.shape=(batch, time_step, vertices, feature)
        A, spatial_attention = args[0], args[1]
        global res
        for i in range(x.shape[1]):
            x_slice = x[:, i, :, :]
            output = self.gnn_layer(x_slice, A, spatial_attention)
            if i == 0:
                res = output.expand_dims(axis=1)
            else:
                res = nd.concat(res, output.expand_dims(axis=1), dim=1)
        return res


# Basic GNN layer
class BasicGNN(HybridBlock):
    def __init__(self, out_units=64, activation='relu', **kwargs):
        super(BasicGNN, self).__init__(**kwargs)
        self.out_units = out_units

        with self.name_scope():
            self.W = self.params.get(
                'W', allow_deferred_init=True
            )
            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)

    def forward(self, X, *args):
        A, spatial_attention = args[0], args[1]
        F = nd
        I = nd.eye(*A.shape)
        A_hat = A.copy() + I
        D = nd.sum(A_hat, axis=0)
        D_inv = D ** -0.5
        D_inv = nd.diag(D_inv)

        A_hat = D_inv * A_hat * D_inv
        if spatial_attention is not None:
            spatial_attention = spatial_attention.transpose((1, 2, 0))
            A_hat = F.dot(A, spatial_attention)
            A_hat = A_hat.transpose((2, 0, 1))
        batch_size, num_of_vertices, num_of_feature = X.shape
        self.W.shape = (num_of_feature, self.out_units)
        self.W._finish_deferred_init()
        if spatial_attention is None:
            X = X.transpose((1, 2, 0))
            aggregate = F.dot(A_hat, X)
            aggregate = aggregate.transpose((2, 0, 1))
        else:
            aggregate = F.batch_dot(A_hat, X)
        propagate = self.activation(
            F.dot(aggregate, self.W.data()))
        return propagate


class gcn_operation(HybridBlock):
    '''
    graph convolutional operation, a simple GCN we defined in paper

    Parameters
    ----------
    data: NDarray, shape is (3N, B, C)

    adj: NDarray, shape is (3N, 3N)

    num_of_filter: int, C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (3N, B, C')

    '''

    def __init__(self, activation, num_of_filter):
        super(gcn_operation, self).__init__()
        assert activation in {'GLU', 'relu'}
        self.activation = activation
        self.num_of_filter = num_of_filter
        with self.name_scope():
            if activation == "GLU":
                self.FC = nn.Dense(units=2 * self.num_of_filter, flatten=False, activation=None)
            elif activation == "relu":
                self.FC = nn.Dense(units=self.num_of_filter, flatten=False, activation=activation)

    def forward(self, x, *args):
        adj = args[0]
        data = x
        # shape is (3N, B, C)
        data = nd.dot(adj, data)
        data = self.FC(data)
        if self.activation == 'GLU':
            # data shape is (3N, B, 2C')
            # shape is (3N, B, C'), (3N, B, C')
            lhs, rhs = nd.split(data, num_outputs=2, axis=2)

            # shape is (3N, B, C')
            return lhs * nd.sigmoid(rhs)

        return data


class stsgcm(HybridBlock):
    '''
    STSGCM, multiple stacked gcn layers with cropping and max operation

    Parameters
    ----------
    data: NDArray, shape is (3N, B, C)

    adj: NDArray, shape is (3N, 3N)

    filters: list[int], list of C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (N, B, C')

    '''

    def __init__(self, num_of_vertices, filters, activation):
        super(stsgcm, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.filters = filters
        self.gcn_list = []
        with self.name_scope():
            for i in range(len(self.filters)):
                self.gcn_list.append(gcn_operation(activation=activation, num_of_filter=filters[i]))
                self.register_child(self.gcn_list[-1])

    def forward(self, x, *args):
        adj = args[0]
        need_concat = []
        _, _, num_of_features = x.shape
        data = x
        for i in range(len(self.filters)):
            data = self.gcn_list[i](data, adj)
            need_concat.append(data)
        # shape of each element is (1, N, B, C')
        need_concat = [
            nd.expand_dims(
                nd.slice(
                    i,
                    begin=(self.num_of_vertices, None, None),
                    end=(2 * self.num_of_vertices, None, None)
                ), 0
            ) for i in need_concat
        ]
        # shape is (N, B, C')
        return nd.max(nd.concat(*need_concat, dim=0), axis=0)


class sthgcn_layer_individual(HybridBlock):
    '''
       STSGCL, multiple individual STSGCMs

       Parameters
       ----------
       data: NDarray, shape is (B, T, N, C)

       adj: NDarray, shape is (3N, 3N)

       T: int, length of time series, T

       num_of_vertices: int, N

       num_of_features: int, C

       filters: list[int], list of C'

       activation: str, {'GLU', 'relu'}

       temporal_emb, spatial_emb: bool

       prefix: str

       Returns
       ----------
       output shape is (B, T-2, N, C')
       '''

    def __init__(self, num_of_vertices, filters, activation, prefix_=""):
        super(sthgcn_layer_individual, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.filters = filters
        self.activation = activation
        self.prefix_ = prefix_
        with self.name_scope():
            self.stsgcm = stsgcm(num_of_vertices, filters, activation)

    def forward(self, x, *args):
        adj = args[0]
        # shape is (B, T, N, C)
        batch_size, T, _, num_of_features = x.shape
        data = x
        need_concat = []
        for i in range(T - 2):
            # shape is (B, 3, N, C)
            t = nd.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

            # shape is (B, 3N, C)
            t = nd.reshape(t, (-1, 3 * self.num_of_vertices, num_of_features))

            # shape is (3N, B, C)
            t = nd.transpose(t, (1, 0, 2))

            # shape is (N, B, C')
            t = self.stsgcm(t, adj)

            # shape is (B, N, C')
            t = nd.swapaxes(t, 0, 1)

            # shape is (B, 1, N, C')
            need_concat.append(nd.expand_dims(t, axis=1))

        # shape is (B, T-2, N, C')
        return nd.concat(*need_concat, dim=1)


class sthgcn_layer_sharing(HybridBlock):
    '''
    STSGCL, multiple a sharing STSGCM

    Parameters
    ----------
    data: NDarray, shape is (B, T, N, C)

    adj: NDarray, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    def __init__(self, num_of_vertices, filters, activation):
        super(sthgcn_layer_sharing, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.filters = filters
        with self.name_scope():
            self.stgcm_layer = stsgcm(num_of_vertices, filters, activation)

    def forward(self, x, *args):
        adj = args[0]
        batch_size, T, _, num_of_features = x.shape
        data = x
        # shape is (B, T, N, C)
        need_concat = []
        for i in range(T - 2):
            # shape is (B, 3, N, C)
            t = nd.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

            # shape is (B, 3N, C)
            t = nd.reshape(t, (-1, 3 * self.num_of_vertices, num_of_features))

            # shape is (3N, B, C)
            t = nd.swapaxes(t, 0, 1)
            need_concat.append(t)

        # shape is (3N, (T-2)*B, C)
        t = nd.concat(*need_concat, dim=1)

        # shape is (N, (T-2)*B, C')
        t = self.stgcm_layer(t, adj)

        # shape is (N, T - 2, B, C)
        t = t.reshape((self.num_of_vertices, T - 2, -1, self.filters[-1]))

        # shape is (B, T - 2, N, C)
        return nd.swapaxes(t, 0, 2)


class cheb_conv_with_SAt(HybridBlock):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''

    def __init__(self, num_of_filters, K, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x, cheb_polynomials, spatial_attention=None, *args, **kwargs):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''

        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        if spatial_attention is None:
            spatial_attention = nd.ones((batch_size, num_of_vertices, num_of_vertices))
        self.Theta.shape = (self.K, num_of_features, self.num_of_filters)
        self.Theta._finish_deferred_init()

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = nd.zeros(shape=(batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta.data()[k]

                # shape is (batch_size, V, F)
                rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),
                                   graph_signal)

                output = output + nd.dot(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return nd.relu(nd.concat(*outputs, dim=-1))


class stsgcl(HybridBlock):
    '''
    STSGCL

    Parameters
    ----------
    data: NDarray, shape is (B, T, N, C)

    adj: NDarray, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    module_type: str, {'sharing', 'individual'}

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    def __init__(self, module_type, num_of_vertices, filters, activation):
        super(stsgcl, self).__init__()
        assert module_type in {'sharing', 'individual'}
        with self.name_scope():
            if module_type == 'individual':
                self.layer = sthgcn_layer_individual(num_of_vertices, filters, activation)
            else:
                self.layer = sthgcn_layer_sharing(num_of_vertices, filters, activation)

    def forward(self, x, *args):
        adj = args[0]
        return self.layer(x, adj)


class FES3(HybridBlock):
    def __init__(self, K, **kwargs):
        '''
        Parameters
        ----------
        config: config context
        '''
        super(FES3, self).__init__(**kwargs)

        num_of_chev_filters = 64
        num_of_time_filters = 64
        time_conv_strides = 1
        with self.name_scope():
            self.cheb_conv_SAt = cheb_conv_with_SAt(
                num_of_filters=num_of_chev_filters,
                K=K)
            self.time_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                strides=(1, time_conv_strides))
            self.residual_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 1),
                strides=(1, time_conv_strides))
            self.ln = nn.LayerNorm(axis=2)

    def forward(self, x, cheb_polynomials, spatial_At, *args, **kwargs):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        mx.ndarray, shape is (batch_size, T_{r-1}, N, num_of_time_filters,)

        '''
        # convert x shape from (batch, time, vertices, feature) into (batch, vertices, features, time)
        x = x.transpose((0, 2, 3, 1))

        spatial_gcn = self.cheb_conv_SAt(x, cheb_polynomials, spatial_At)

        # convolution along time axis
        time_conv_output = (self.time_conv(spatial_gcn.transpose((0, 2, 1, 3)))
                            .transpose((0, 2, 1, 3)))

        # residual shortcut
        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))

        x = self.ln(nd.relu(x_residual + time_conv_output))
        # convert x shape from (batch, vertices, features, time) into (batch, time, vertices, feature)
        x = x.transpose((0, 3, 1, 2))
        return x


class FES4(HybridBlock):
    def __init__(self, num_of_vertices, filter_list, module_type, activation, prefix_=""):
        super(HybridBlock, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.filter_list = filter_list
        self.module_type = module_type
        self.activation = activation
        self.prefix_ = prefix_
        self.stsgcl_layers = []
        with self.name_scope():
            for _ in self.filter_list:
                self.stsgcl_layers.append(
                    stsgcl(module_type, num_of_vertices, filter_list, activation)
                )
                self.register_child(self.stsgcl_layers[-1])

    def forward(self, x, adj, *arg):
        if adj.shape[0] == self.num_of_vertices:
            adj = construct_adj(adj, 3)
        elif adj.shape[0] != 3 * self.num_of_vertices:
            raise Exception(f"adj shape:{adj.shape} error")
        for idx, filters in enumerate(self.filter_list):
            x = self.stsgcl_layers[idx](x, adj)
        return x


class OS1(HybridBlock):
    def __init__(self, num_of_vertices, batch_size):
        super(OS1, self).__init__()
        self.hidden_layer_specs = [(64, 'tanh'), (64, 'tanh')]  # Format: (units in layer, activation function)
        self.num_frame_decoder = 12
        self.batch_size = batch_size
        with self.name_scope():
            self.encoder_lstm_cell = DeferredInitLSTMCell(num_of_vertices)
            # 此处只有一个LSTM cell 前向传播时循环多次输出预测的多个时间帧
            self.decoder = mx.gluon.rnn.LSTMCell(num_of_vertices, input_size=num_of_vertices)

    def forward(self, x, *args):
        global res
        for i in range(x.shape[1]):
            x_slice = x[:, i, :, :]
            if i == 0:
                begin_state = self.encoder_lstm_cell.begin_state(batch_size=self.batch_size, func=nd.random.normal)
                S_h, S_c = self.encoder_lstm_cell(x_slice, begin_state)
            else:
                S_c[0] = nd.array(S_c[0])
                S_c[1] = nd.array(S_c[1])
                S_h, S_c = self.encoder_lstm_cell(x_slice, S_c)
        for i in range(self.num_frame_decoder):
            S_c[0] = nd.array(S_c[0])
            S_c[1] = nd.array(S_c[1])
            S_h, S_c = self.decoder(S_h, S_c)
            if i == 0:
                res = S_h.expand_dims(axis=1)
            else:
                res = nd.concat(res, S_h.expand_dims(axis=1), dim=1)
        return res


class output_layer(HybridBlock):
    '''
    Parameters
    ----------
    data: NDarray, shape is (B, T, N, C)

    num_of_vertices: int, N

    input_length: int, length of time series, T

    num_of_features: int, C

    num_of_filters: int, C'

    predict_length: int, length of predicted time series, T'

    Returns
    ----------
    output shape is (B, T', N)
    '''

    def __init__(self, num_of_vertices, num_of_filters=128, predict_length=12):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        with self.name_scope():
            self.dense_layer1 = nn.Dense(units=self.num_of_filters, flatten=False, activation="relu")
            self.dense_layer2 = nn.Dense(units=self.predict_length, flatten=False, activation=None)

    def forward(self, x, *args):
        batch_size, input_length, _, num_of_features = x.shape
        # data shape is (B, N, T, C)
        data = nd.swapaxes(x, 1, 2)

        # (B, N, T * C)
        data = nd.reshape(
            data, (-1, self.num_of_vertices, input_length * num_of_features)
        )

        # (B, N, C')
        data = self.dense_layer1(data)

        # (B, N, T')
        data = self.dense_layer2(data)

        # (B, T', N)
        data = nd.swapaxes(data, 1, 2)

        return data


class OS2(HybridBlock):
    '''
    a module in ASTGCN
    '''

    def __init__(self, filter_size, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(OS2, self).__init__(**kwargs)
        with self.name_scope():
            # use convolution to generate the prediction
            # instead of using the fully connected layer
            self.final_conv = nn.Conv2D(
                channels=12,
                kernel_size=(1, filter_size))
            self.W = self.params.get("W", allow_deferred_init=True)

    def forward(self, x, *args, **kwargs):
        '''
        Parameters
        ----------
        x: mx.ndarray,
           shape is (batch_size, num_of_timesteps, num_of_vertices, num_of_features)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_for_prediction, num_of_vertices)

        '''
        # (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
        module_output = (self.final_conv(x)
                         [:, :, :, -1].transpose((0, 2, 1)))
        _, num_of_vertices, num_for_prediction = module_output.shape
        self.W.shape = (num_of_vertices, num_for_prediction)
        self.W._finish_deferred_init()
        # (batch_size, num_of_vertices, num_for_prediction)
        x = module_output * self.W.data()
        x = x.transpose((0, 2, 1))
        return x


class OS3(HybridBlock):
    def __init__(self, num_of_vertices):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.output_layers = []
        with self.name_scope():
            for i in range(12):
                self.output_layers.append(output_layer(num_of_vertices, predict_length=1))
                self.register_child(self.output_layers[-1])

    def forward(self, x, *args, **kwargs):
        need_concat = []
        for i in range(12):
            need_concat.append(
                self.output_layers[i](x)
            )
        data = nd.concat(*need_concat, dim=1)
        return data


class DeferredInitLSTMCell(HybridRecurrentCell):
    r"""Long-Short Term Memory (LSTM) network cell.

    Each call computes the following function:

    .. math::
        \begin{array}{ll}
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
        o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        h_t = o_t * \tanh(c_t)
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the hidden state of the previous
    layer at time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer, default 'zeros'
        Initializer for the bias vector.
    prefix : str, default ``'lstm_'``
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None, default None
        Container for weight sharing between cells.
        Created if `None`.
    activation : str, default 'tanh'
        Activation type to use. See nd/symbol Activation
        for supported types.
    recurrent_activation : str, default 'sigmoid'
        Activation type to use for the recurrent step. See nd/symbol Activation
        for supported types.

    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of two initial recurrent state tensors. Each has shape
          `(batch_size, num_hidden)`.

    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of two output recurrent state tensors. Each has
          the same shape as `states`.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None, activation='tanh',
                 recurrent_activation='sigmoid'):
        super(DeferredInitLSTMCell, self).__init__(prefix=prefix, params=params)

        self._hidden_size = hidden_size
        self._input_size = input_size
        self.i2h_weight = self.params.get('i2h_weight',
                                          init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=(4 * hidden_size, hidden_size),
                                          init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(4 * hidden_size,),
                                        init=i2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(4 * hidden_size,),
                                        init=h2h_bias_initializer,
                                        allow_deferred_init=True)
        self._activation = activation
        self._recurrent_activation = recurrent_activation

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'lstm'

    def __repr__(self):
        s = '{name}({mapping})'
        shape = self.i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def forward(self, inputs, states):
        F = nd
        self.i2h_weight.shape = (4 * self._hidden_size, inputs.shape[-1] * inputs.shape[-2])
        self.i2h_weight._finish_deferred_init()
        # pylint: disable=too-many-locals
        prefix = 't%d_' % self._counter
        i2h = F.FullyConnected(data=inputs, weight=self.i2h_weight.data(), bias=self.i2h_bias.data(),
                               num_hidden=self._hidden_size * 4, name=prefix + 'i2h')
        h2h = F.FullyConnected(data=states[0], weight=self.h2h_weight.data(), bias=self.h2h_bias.data(),
                               num_hidden=self._hidden_size * 4, name=prefix + 'h2h')
        gates = F.elemwise_add(i2h, h2h, name=prefix + 'plus0')
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix + 'slice')
        in_gate = self._get_activation(
            F, slice_gates[0], self._recurrent_activation, name=prefix + 'i')
        forget_gate = self._get_activation(
            F, slice_gates[1], self._recurrent_activation, name=prefix + 'f')
        in_transform = self._get_activation(
            F, slice_gates[2], self._activation, name=prefix + 'c')
        out_gate = self._get_activation(
            F, slice_gates[3], self._recurrent_activation, name=prefix + 'o')
        next_c = F.elemwise_add(F.elemwise_mul(forget_gate, states[1], name=prefix + 'mul0'),
                                F.elemwise_mul(in_gate, in_transform, name=prefix + 'mul1'),
                                name=prefix + 'state')
        next_h = F.elemwise_mul(out_gate, F.Activation(next_c, act_type=self._activation, name=prefix + 'activation0'),
                                name=prefix + 'out')

        return next_h, [next_h, next_c]


class MOBF1(HybridBlock):
    def __init__(self, num_of_data, num_of_vertices):
        super(MOBF1, self).__init__()
        self.param_list = []
        with self.name_scope():
            self.MOBF1_embedding = MOBFEmbedding(num_of_data, num_of_vertices, True)
            for i in range(num_of_data):
                self.param_list.append(self.params.get('W_' + str(i), allow_deferred_init=True))

    def forward(self, end_data_list, output_data_list, *args, **kwargs):
        end_data_list = self.MOBF1_embedding(end_data_list, output_data_list)
        global data
        for i, x in enumerate(end_data_list):
            batch_size = x.shape[0]
            # batch_dim = 1
            self.param_list[i].shape = (1,) + x.shape[1:]
            self.param_list[i]._finish_deferred_init()
            # copy on batch_dim
            temp = nd.repeat(self.param_list[i].data(), batch_size, 0) * x
            if i == 0:
                data = temp
            else:
                data = data + temp
        return data


class MOBFEmbedding(HybridBlock):
    def __init__(self, data_list_len, num_of_vertices, temporal_embedding):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.temporal_dense_layers = []
        self.feature_dense_layers = []
        self.temporal_embedding = temporal_embedding
        with self.name_scope():
            for i in range(data_list_len):
                # (batch_size, time_step, vertices, feature)
                if temporal_embedding:
                    self.temporal_dense_layers.append(mx.gluon.nn.Dense(12, activation='relu'))
                    self.register_child(self.temporal_dense_layers[-1])
                self.feature_dense_layers.append(mx.gluon.nn.Dense(64, activation='relu'))
                self.register_child(self.feature_dense_layers[-1])

    def forward(self, end_data_list, output_data_list, *args, **kwargs):
        data = []
        for i in range(len(end_data_list)):
            x = output_data_list[end_data_list[i]]
            batch_size, time_step, _, feature = x.shape
            x = x.reshape(-1, x.shape[-1])
            x = self.feature_dense_layers[i](x).reshape(batch_size, time_step, self.num_of_vertices, 64)
            # (batch, time_step, num_of_vertices, feature)->(batch, num_of_vertices, feature, time_step)->(-1, time_step)
            if self.temporal_embedding:
                x = x.transpose((0, 2, 3, 1)).reshape(-1, time_step)
                data.append(self.temporal_dense_layers[i](x).reshape(-1, self.num_of_vertices, 64, 12).transpose(
                    (0, 3, 1, 2)))
            else:
                data.append(x)
        return data
