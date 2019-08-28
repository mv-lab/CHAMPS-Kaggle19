import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common import *
from torch_scatter import *
from torch_geometric.utils import scatter_



#############################################################################
CONVERSION=[
 'preprocess.0.linear.weight',	 'preprocess.0.linear.weight',
 'preprocess.0.bn.weight',	 'preprocess.0.bn.weight',
 'preprocess.0.bn.bias',	 'preprocess.0.bn.bias',
 'preprocess.0.bn.running_mean',	 'preprocess.0.bn.running_mean',
 'preprocess.0.bn.running_var',	 'preprocess.0.bn.running_var',
 'preprocess.2.linear.weight',	 'preprocess.2.linear.weight',
 'preprocess.2.bn.weight',	 'preprocess.2.bn.weight',
 'preprocess.2.bn.bias',	 'preprocess.2.bn.bias',
 'preprocess.2.bn.running_mean',	 'preprocess.2.bn.running_mean',
 'preprocess.2.bn.running_var',	 'preprocess.2.bn.running_var',
 'propagate.bias',	 'conv.bias',
 'propagate.encoder.0.linear.weight',	 'conv.nn.0.linear.weight',
 'propagate.encoder.0.bn.weight',	 'conv.nn.0.bn.weight',
 'propagate.encoder.0.bn.bias',	 'conv.nn.0.bn.bias',
 'propagate.encoder.0.bn.running_mean',	 'conv.nn.0.bn.running_mean',
 'propagate.encoder.0.bn.running_var',	 'conv.nn.0.bn.running_var',
 'propagate.encoder.2.linear.weight',	 'conv.nn.2.linear.weight',
 'propagate.encoder.2.bn.weight',	 'conv.nn.2.bn.weight',
 'propagate.encoder.2.bn.bias',	 'conv.nn.2.bn.bias',
 'propagate.encoder.2.bn.running_mean',	 'conv.nn.2.bn.running_mean',
 'propagate.encoder.2.bn.running_var',	 'conv.nn.2.bn.running_var',
 'propagate.encoder.4.linear.weight',	 'conv.nn.4.linear.weight',
 'propagate.encoder.4.bn.weight',	 'conv.nn.4.bn.weight',
 'propagate.encoder.4.bn.bias',	 'conv.nn.4.bn.bias',
 'propagate.encoder.4.bn.running_mean',	 'conv.nn.4.bn.running_mean',
 'propagate.encoder.4.bn.running_var',	 'conv.nn.4.bn.running_var',
 'propagate.encoder.6.linear.weight',	 'conv.nn.6.linear.weight',
 'propagate.encoder.6.bn.weight',	 'conv.nn.6.bn.weight',
 'propagate.encoder.6.bn.bias',	 'conv.nn.6.bn.bias',
 'propagate.encoder.6.bn.running_mean',	 'conv.nn.6.bn.running_mean',
 'propagate.encoder.6.bn.running_var',	 'conv.nn.6.bn.running_var',
 'propagate.gru.weight_ih_l0',	 'gru.weight_ih_l0',
 'propagate.gru.weight_hh_l0',	 'gru.weight_hh_l0',
 'propagate.gru.bias_ih_l0',	 'gru.bias_ih_l0',
 'propagate.gru.bias_hh_l0',	 'gru.bias_hh_l0',
 'set2set.lstm.weight_ih_l0',	 'set2set.lstm.weight_ih_l0',
 'set2set.lstm.weight_hh_l0',	 'set2set.lstm.weight_hh_l0',
 'set2set.lstm.bias_ih_l0',	 'set2set.lstm.bias_ih_l0',
 'set2set.lstm.bias_hh_l0',	 'set2set.lstm.bias_hh_l0',
 'predict.0.linear.weight',	 'predict.0.linear.weight',
 'predict.0.bn.weight',	 'predict.0.bn.weight',
 'predict.0.bn.bias',	 'predict.0.bn.bias',
 'predict.0.bn.running_mean',	 'predict.0.bn.running_mean',
 'predict.0.bn.running_var',	 'predict.0.bn.running_var',
 'predict.2.linear.weight',	 'predict.2.linear.weight',
 'predict.2.bn.weight',	 'predict.2.bn.weight',
 'predict.2.bn.bias',	 'predict.2.bn.bias',
 'predict.2.bn.running_mean',	 'predict.2.bn.running_mean',
 'predict.2.bn.running_var',	 'predict.2.bn.running_var',
 'predict.4.weight',	 'predict.4.weight',
 'predict.4.bias',	 'predict.4.bias',
]


def load_pretrain(net, pretrain_file, conversion=CONVERSION):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1,2)
    for key, pretrain_key in conversion:
        if any(s in key for s in []):
            continue

        #print('\t\t',key)
        print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
            key, str(state_dict[key].shape),
            pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
        ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')







#############################################################################
class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv, self).__init__()

        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim),
            #nn.ReLU(inplace=True),
        )

        self.gru  = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))


    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()

        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = self.encoder(edge).view(-1,node_dim,node_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)
        message = x_i.view(-1,1,node_dim)@edge
        message = message.view(-1,node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)
        message = F.relu(message +self.bias)

        #2. update: n_j = f(n_j, m_j)
        update = message

        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,node_dim), hidden)
        update = update.view(-1,node_dim)

        return update, hidden

class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star



#message passing
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

        self.propagate = GraphConv(128, edge_dim)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)


        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(4*128, 1024),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn( 1024, 512),
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

        pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool,node0,node1],-1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict



# def criterion(predict, coupling_value):
#     predict = predict.view(-1)
#     coupling_value = coupling_value.view(-1)
#     assert(predict.shape==coupling_value.shape)
#
#     loss = F.mse_loss(predict, coupling_value)
#     return loss

def criterion(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss


##################################################################################################################

def make_dummy_data(node_dim, edge_dim, num_target, batch_size):

    #dummy data
    num_node = []
    num_edge = []

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []


    for b in range(batch_size):
        node_offset = sum(num_node)
        edge_offset = sum(num_edge)

        N = np.random.choice(10)+8
        E = np.random.choice(10)+16
        node.append(np.random.uniform(-1,1,(N,node_dim)))
        edge.append(np.random.uniform(-1,1,(E,edge_dim)))

        edge_index.append(np.random.choice(N, (E,2))+node_offset)
        node_index.append(np.array([b]*N))

        #---
        C = np.random.choice(10)+1
        coupling_value.append(np.random.uniform(-1,1, C))
        coupling_atom_index.append(np.random.choice(N,(C,2))+node_offset)
        coupling_type_index.append(np.random.choice(num_target, C))
        coupling_batch_index.append(np.array([b]*C))

        #---
        num_node.append(N)
        num_edge.append(E)


    node = torch.from_numpy(np.concatenate(node)).float().cuda()
    edge = torch.from_numpy(np.concatenate(edge)).float().cuda()
    edge_index = torch.from_numpy(np.concatenate(edge_index)).long().cuda()
    node_index = torch.from_numpy(np.concatenate(node_index)).long().cuda()

    #---
    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float().cuda()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(np.array(coupling_index)).long().cuda()


    return node, edge, edge_index, node_index, coupling_value, coupling_index



def run_check_net():

    #dummy data
    node_dim = 5
    edge_dim = 7
    num_target = 8
    batch_size = 16
    node, edge, edge_index, node_index, coupling_value, coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)

    print('batch_size ', batch_size)
    print('----')
    print('node',node.shape)
    print('edge',edge.shape)
    print('edge_index',edge_index.shape)
    print('node_index',node_index.shape)
    print('----')

    print('coupling_index',coupling_index.shape)
    print('')

    #---
    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()



    predict = net(node, edge, edge_index, node_index, coupling_index)

    print('predict: ', predict.shape)
    print(predict)
    print('')

    if 0:
        keys = list(net.state_dict().keys())
        sorted(keys)
        for k in keys:
            if '.num_batches_tracked' in k:
                continue
            print(' \'%s\','%k)



def run_check_net2():

    #dummy data
    node_dim = 93
    edge_dim = 6
    num_target = 8

    data = read_pickle_from_file( '/root/share/project/kaggle/2019/champs_scalar/result/backup/data.pickle' )

    node = data['node'].cuda()
    edge = data['edge'].cuda()
    edge_index = data['edge_index'].cuda()
    node_index = data['node_batch_index'].cuda()
    coupling_index = data['coupling_index'].cuda()
    coupling_type = data['coupling_type'].cuda()
    coupling_value = data['coupling_value'].cuda()
    coupling_batch_index = data['coupling_batch_index'].cuda()
    infor = data['infor']

    coupling_index = torch.cat([
        coupling_index,
        coupling_type.view(-1,1),
        coupling_batch_index.view(-1,1)
    ],-1)



    #---
    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()

    pretrain_file = '/root/share/project/kaggle/2019/champs_scalar/result/backup/00370000_model.pth'
    load_pretrain(net,pretrain_file)

    predict = net(node, edge, edge_index, node_index, coupling_index)

    print('predict: ', predict.shape)
    print(predict)
    print('')
    #
    # if 0:
    #     keys = list(net.state_dict().keys())
    #     sorted(keys)
    #     for k in keys:
    #         if '.num_batches_tracked' in k:
    #             continue
    #         print(' \'%s\','%k)

    torch.save(net.state_dict(), '/root/share/project/kaggle/2019/champs_scalar/result/backup/00370000_model.convert.pth')


'''
tensor([ 83.2995,  -2.4076,   7.0543,  ...,  22.8560,   0.1970, 124.0506],
       device='cuda:0', grad_fn=<ViewBackward>)
'''

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net2()


    print('\nsucess!')

