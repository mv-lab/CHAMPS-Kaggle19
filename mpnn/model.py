import os
from torch_geometric.utils import scatter_
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import *

from common import *

from lib.net.lstm_norm import LSTM, BNLSTMCell
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def l2_loss(predict, coupling_value):
    predict = predict.view(-1)
    coupling_value = coupling_value.view(-1)
    assert(predict.shape == coupling_value.shape)

    loss = F.mse_loss(predict, coupling_value)
    return loss

def log_l1_loss(predict, truth):
    predict = predict.view(-1)
    truth = truth.view(-1)
    assert(predict.shape == truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss

def huber_loss(predict, truth):
    predict = predict.view(-1)
    truth = truth.view(-1)
    assert(predict.shape == truth.shape)

    loss = torch.F.smooth_l1_loss(predict, truth)
    return loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        torch.nn.init.zeros_(m.bias)


class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-05, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GraphConvGru(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GraphConvGru, self).__init__()       
        self.node_dim = node_dim

        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim)) # nn.ReLU(inplace=True),

        self.gru = nn.GRU(node_dim, node_dim,
                          batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim),
                                1.0 / math.sqrt(node_dim))

    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        # print(node.shape)
        # print(edge.shape)
        edge_index = edge_index.t().contiguous()

        # 1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i = torch.index_select(node, 0, edge_index[0])

        edge = self.encoder(edge).view(-1, node_dim, node_dim)

        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)

        message = x_i.view(-1, 1, node_dim)@edge
        message = message.view(-1, node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)

        message = F.relu(message + self.bias)

        # 2. update: n_j = f(n_j, m_j)
        update = message

        # batch_first=True
        update, hidden = self.gru(update.view(1, -1, node_dim), hidden)
        update = update.view(-1, node_dim)

        return update, hidden

    def reset_parameters(self):
        self.bias = nn.Parameter(torch.Tensor(self.node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.node_dim),
                                1.0 / math.sqrt(self.node_dim))


class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GraphConv, self).__init__()       
        self.node_dim = node_dim

        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim)) # nn.ReLU(inplace=True),

        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim),
                                1.0 / math.sqrt(node_dim))

    def forward(self, node, edge_index, edge):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        # print(node.shape)
        # print(edge.shape)
        edge_index = edge_index.t().contiguous()

        # 1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i = torch.index_select(node, 0, edge_index[0])
        
        edge = self.encoder(edge).view(-1, node_dim, node_dim)

        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)

        message = x_i.view(-1, 1, node_dim)@edge
        message = message.view(-1, node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)

        message = F.relu(message + self.bias)

        return message

    def reset_parameters(self):
        self.bias = nn.Parameter(torch.Tensor(self.node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.node_dim),
                                1.0 / math.sqrt(self.node_dim))


class GraphConvSag(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GraphConvSag, self).__init__()       
        self.node_dim = node_dim

        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim)) # nn.ReLU(inplace=True),

        self.predict = nn.Sequential(
            LinearBn(node_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 1))

        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim),
                                1.0 / math.sqrt(node_dim))

    def forward(self, node, edge_index, edge):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        # print(node.shape)
        # print(edge.shape)
        edge_index = edge_index.t().contiguous()

        # 1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i = torch.index_select(node, 0, edge_index[0])
        
        edge = self.encoder(edge).view(-1, node_dim, node_dim)

        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)

        message = x_i.view(-1, 1, node_dim)@edge
        message = message.view(-1, node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)

        message = F.relu(message + self.bias)

        score = self.predict(message)

        return score

    def reset_parameters(self):
        self.bias = nn.Parameter(torch.Tensor(self.node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.node_dim),
                                1.0 / math.sqrt(self.node_dim))


class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        hidden_size = in_channel
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layer = num_layer
        self.lstm = LSTM(BNLSTMCell, in_channel, hidden_size=hidden_size, out_size=out_channel, 
                        num_layers=num_layer, max_length=512)
        #self.lstm = LayerNormLSTM(out_channel, in_channel, num_layer, dropout=0.1)
        #self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        #self.lstm.reset_parameters()


    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1
        h = (x.new_zeros((batch_size, self.in_channel)),
             x.new_zeros((batch_size, self.in_channel)))
        q_star = x.new_zeros(batch_size, self.out_channel)

        for i in range(self.processing_step):     
            #print('hshape: ' + str(h[0].shape))    
            #print('q_star shape: ' + str(q_star.shape))  
            q, h = self.lstm(q_star.unsqueeze(0), hx=h) 

            if len(h[0].size()) > 2:
                h = (torch.squeeze(h[0], dim=0), torch.squeeze(h[1], dim=0))

            q = q.view(batch_size, -1)
            e = (x * q[batch_index]).sum(dim=-1, keepdim=True)  # shape = num_node x 1
            # shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)
            # apply attention #shape = batch_size x ...
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
        return q_star


class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(Net, self).__init__()
        self.num_propagate = 6
        self.num_s2s = 6

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConvGru(128, edge_dim)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)

        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(512, 1024),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):

        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node   = self.preprocess(node)
        hidden = node.view(1,num_node,-1)

        for i in range(self.num_propagate):
            node, hidden =  self.propagate(node, edge_index, edge, hidden)

        pool = self.set2set(node, node_index)

        #---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)

        pool  = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool,node0,node1],-1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict


# message passing
class Set2SetLargerNet(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(Set2SetLargerNet, self).__init__()
        self.num_propagate = 1
        self.num_s2s = 1

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 512),
            nn.ReLU(inplace=True),
            LinearBn(512, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConvGru(256, edge_dim)
        self.set2set = Set2Set(256, processing_step=self.num_s2s)

        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(1024, 2048),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn(2048, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node = self.preprocess(node)
        hidden = node.view(1, num_node, -1)
        for i in range(self.num_propagate):
            node, hidden = self.propagate(node, edge_index, edge, hidden)

        pool = self.set2set(node, node_index)

        # ---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index, 1, dim=1)

        pool = torch.index_select(
            pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select(
            node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select(
            node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool, node0, node1], -1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict


class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConvGru`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConvGru`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """

    def __init__(self, in_channels, out_channel, ratio=0.5, GNN=GraphConvSag, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, out_channel, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()


    def forward(self, x, edge_index, edge=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(x, edge_index, edge)
        #print(score.view(-1).shape)
        score = score.view(-1)
        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index = edge_index.t().contiguous()
        #print(edge_index.shape)
        #print(perm.shape)
        edge_index, edge = filter_adj(edge_index, edge, perm, num_nodes=score.size(0))

        return x, edge_index, edge, batch, perm, score[perm]


    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)


class SagPoolNet(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(SagPoolNet, self).__init__()

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConvGru(128, edge_dim)
        self.graph_conv = GraphConv(128, edge_dim)
        self.sag_pool = SAGPooling(128, edge_dim, ratio=0.9)

        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(512, 1024),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node = self.preprocess(node)
        hidden = node.view(1, num_node, -1)

        node, hidden = self.propagate(node, edge_index, edge, hidden)
        pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        pool1 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)
        
        node = self.graph_conv(node, edge_index, edge)
        pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        pool2 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)

        #node = self.graph_conv(node, edge_index, edge)
        #pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        #pool3 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)

        pool = pool1 + pool2 # + pool3

        # ---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index, 1, dim=1)

        pool = torch.index_select(
            pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select(
            node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select(
            node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool, node0, node1], -1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict


class SagPoolLargerNet(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(SagPoolLargerNet, self).__init__()

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 512),
            nn.ReLU(inplace=True),
            LinearBn(512, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConvGru(256, edge_dim)
        self.graph_conv = GraphConv(256, edge_dim)
        self.sag_pool = SAGPooling(256, edge_dim, ratio=0.9)

        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(1024, 2048),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn(2048, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node = self.preprocess(node)
        hidden = node.view(1, num_node, -1)
        
        node, hidden = self.propagate(node, edge_index, edge, hidden)
        
        pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        pool1 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)
        
        node = self.graph_conv(node, edge_index, edge)
        pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        pool2 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)

        #node = self.graph_conv(node, edge_index, edge)
        #pool, _, _, batch, _, _ = self.sag_pool(node, edge_index, edge, node_index)
        #pool3 = torch.cat([gmp(pool, batch), gap(pool, batch)], dim=1)

        pool = pool1 + pool2 #+ pool3

        # ---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index, 1, dim=1)

        pool = torch.index_select(
            pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select(
            node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select(
            node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool, node0, node1], -1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict


##################################################################################################################

def make_dummy_data(node_dim, edge_dim, num_target, batch_size):

    # dummy data
    num_node = []
    num_edge = []

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index = []
    coupling_type_index = []
    coupling_batch_index = []

    for b in range(batch_size):
        node_offset = sum(num_node)
        edge_offset = sum(num_edge)

        N = np.random.choice(10)+8
        E = np.random.choice(10)+16
        node.append(np.random.uniform(-1, 1, (N, node_dim)))
        edge.append(np.random.uniform(-1, 1, (E, edge_dim)))

        edge_index.append(np.random.choice(N, (E, 2))+node_offset)
        node_index.append(np.array([b]*N))

        # ---
        C = np.random.choice(10)+1
        coupling_value.append(np.random.uniform(-1, 1, C))
        coupling_atom_index.append(np.random.choice(N, (C, 2))+node_offset)
        coupling_type_index.append(np.random.choice(num_target, C))
        coupling_batch_index.append(np.array([b]*C))

        # ---
        num_node.append(N)
        num_edge.append(E)

    node = torch.from_numpy(np.concatenate(node)).float().cuda()
    edge = torch.from_numpy(np.concatenate(edge)).float().cuda()
    edge_index = torch.from_numpy(np.concatenate(edge_index)).long().cuda()
    node_index = torch.from_numpy(np.concatenate(node_index)).long().cuda()

    # ---
    coupling_value = torch.from_numpy(
        np.concatenate(coupling_value)).float().cuda()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1, 1),
        np.concatenate(coupling_batch_index).reshape(-1, 1),
    ], -1)
    coupling_index = torch.from_numpy(np.array(coupling_index)).long().cuda()

    return node, edge, edge_index, node_index, coupling_value, coupling_index


def run_check_net():
    # dummy data
    node_dim = 5
    edge_dim = 7
    num_target = 8
    batch_size = 16
    node, edge, edge_index, node_index, coupling_value, coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)

    print('batch_size ', batch_size)
    print('----')
    print('node', node.shape)
    print('edge', edge.shape)
    print('edge_index', edge_index.shape)
    print('node_index', node_index.shape)
    print('----')

    print('coupling_index', coupling_index.shape)
    print('')

    # ---
    net = Net(node_dim=node_dim, edge_dim=edge_dim,
              num_target=num_target).cuda()
    net = net.eval()

    predict = net(node, edge, edge_index, node_index, coupling_index)

    print('predict: ', predict.shape)
    print(predict)
    print('')

    #keys = list(net.state_dict().keys())
    # sorted(keys)
    # for k in keys:
    #    if '.num_batches_tracked' in k:
    #        continue
    #    print(' \'%s\','%k)


def run_check_train():
    node_dim = 15
    edge_dim = 5
    num_target = 12
    batch_size = 64
    node, edge, edge_index, node_index, coupling_value, coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)

    net = Net(node_dim=node_dim, edge_dim=edge_dim,
              num_target=num_target).cuda()
    net = net.eval()

    predict = net(node, edge, edge_index, node_index, coupling_index)
    loss = log_l1_loss(predict, coupling_value)

    print('*loss = %0.5f' % (loss.item(),))
    print('')

    print('predict: ', predict.shape)
    print(predict)
    print(coupling_value)
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)

    print('--------------------')
    print('[iter ]  loss       ')
    print('--------------------')

    i = 0
    optimizer.zero_grad()
    while i <= 500:
        net.train()
        optimizer.zero_grad()

        predict = net(node, edge, edge_index, node_index, coupling_index)
        loss = log_l1_loss(predict, coupling_value)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[%05d] %8.5f  ' % (
                i,
                loss.item(),
            ))
        i = i+1
    print('')

    # check results
    print(predict[:5])
    print(coupling_value[:5])
    print('')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()
    # run_check_train()

    print('\nsuccess!')
