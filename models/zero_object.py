from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.net_util import norm_col_init, weights_init, toFloatTensor
import scipy.sparse as sp
import numpy as np
from datasets.glove import Glove
from .model_io import ModelOutput
from utils import flag_parser
from datasets.constants import UNSEEN_FULL_OBJECT_8CLASS_LIST,UNSEEN_FULL_OBJECT_4CLASS_LIST
import random
args = flag_parser.parse_arguments()


class SelfAttention(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs):
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)

        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)

        out = torch.matmul(alpha, V)

        return out


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).tocoo()


class ZeroZero(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(ZeroZero, self).__init__()
        self.embed_action = nn.Linear(action_space, 10)

        n = 20
        self.unseen_objects=[]
        lstm_input_sz = 6 + n * 5

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

        # glove embeddings for all the objs.
        # self.objects = []
        # with open("/home/Newdisk/lixinting/zero-shot-data/data/gcn/objects.txt") as f: # 至少这里
        #     objects = f.readlines()
        #     for o in objects:
        #         o = o.strip()
        #         self.objects.append(o)
        
        self.n = n
        # self.glove = Glove(args.glove_file)
        
        self.selfatt = SelfAttention(5, False)

    def list_from_raw_obj(self, objbb, target, target_name):
        glove = Glove(args.glove_file)
        objstate = torch.zeros(101, 5)
        cos = torch.nn.CosineSimilarity(dim=1)
        # glove_sim = cos(self.all_glove.detach(), target[None, :])[:, None]
        
        j = 1
        for k, v in objbb.items():
            objstate[j][0] = 1 if k == target_name else 0
            object_cos = nn.Parameter(torch.Tensor(glove.glove_embeddings[k][:]))
            object_cos.requires_grad = False
            object_cos = object_cos.to('cuda:'+str(self.gpu_id))
            objstate[j][1] = cos(object_cos.detach(), target[None, :])[:, None]
            x1 = v[0::4]
            y1 = v[1::4]
            x2 = v[2::4]
            y2 = v[3::4]
            objstate[j][2] = np.sum(x1 + x2) / len(x1 + x2) / 300
            objstate[j][3] = np.sum(y1 + y2) / len(y1 + y2) / 300
            objstate[j][4] = abs(max(x2) - min(x1)) * abs(max(y2) - min(y1)) / 300 / 300
            j = j+1
        if args.gpu_ids != -1:
            objstate = objstate.cuda()
        # objstate = torch.cat((objstate, glove_sim), dim=1)
        # 排序
        tmp = objstate[:,1:2]
        index = torch.argsort(tmp.squeeze(),descending=True)
        objstate = torch.index_select(objstate, 0, index=index.squeeze())
        objstate = objstate[0:self.n][:]
        return objstate

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        objbb = model_input.objbb
        (hx, cx) = model_input.hidden
        self.gpu_id = model_input.gpu_id
        target = model_input.target_class_embedding
        target_name = model_input.target_name
        action_probs = model_input.action_probs
        objstate = self.list_from_raw_obj(objbb, target, target_name).unsqueeze(0)
        x = self.selfatt(objstate).view(1, -1)
        # action_embedding = F.relu(self.embed_action(action_probs))
        x = torch.cat((x, action_probs), dim=1)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        image_embedding = None
        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )

    # def random_self_objects(self):
    #     random.shuffle(self.objects)

    def order_self_objects(self):
        pass


class ZeroGCN(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(ZeroGCN, self).__init__()
        self.embed_action = nn.Linear(action_space, 10)

        n = 20
        feature = 5
        self.unseen_objects=[]
        lstm_input_sz = 6 + n * feature

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.graph_detection_other_info_linear_1 = nn.Linear(n, n)
        self.graph_detection_other_info_linear_2 = nn.Linear(n, n)
        self.graph_detection_other_info_linear_3 = nn.Linear(n, n)
        self.graph_detection_other_info_linear_4 = nn.Linear(n, n)
        self.graph_detection_other_info_linear_5 = nn.Linear(n, n)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

        self.n = n
        # self.glove = Glove(args.glove_file)
        self.selfatt = SelfAttention(5, False)

    def list_from_raw_obj(self, objbb, target, target_name):
        glove = Glove(args.glove_file)
        objstate = torch.zeros(101, 5)
        cos = torch.nn.CosineSimilarity(dim=1)
        # glove_sim = cos(self.all_glove.detach(), target[None, :])[:, None]
        
        j = 1
        for k, v in objbb.items():
            objstate[j][0] = 1 if k == target_name else 0
            object_cos = nn.Parameter(torch.Tensor(glove.glove_embeddings[k][:]))
            object_cos.requires_grad = False
            object_cos = object_cos.to('cuda:'+str(self.gpu_id))
            objstate[j][1] = cos(object_cos.detach(), target[None, :])[:, None]
            x1 = v[0::4]
            y1 = v[1::4]
            x2 = v[2::4]
            y2 = v[3::4]
            objstate[j][2] = np.sum(x1 + x2) / len(x1 + x2) / 300
            objstate[j][3] = np.sum(y1 + y2) / len(y1 + y2) / 300
            objstate[j][4] = abs(max(x2) - min(x1)) * abs(max(y2) - min(y1)) / 300 / 300
            j = j+1
        if args.gpu_ids != -1:
            objstate = objstate.to('cuda:'+str(self.gpu_id))
        # objstate = torch.cat((objstate, glove_sim), dim=1)
        # 排序
        tmp = objstate[:,1:2]
        index = torch.argsort(tmp.squeeze(),descending=True)
        objstate = torch.index_select(objstate, 0, index=index.squeeze())
        objstate = objstate[0:self.n][:]
        return objstate

    def gcn(self, objstate):
        objstate = objstate.t()
        object_info = F.relu(self.graph_detection_other_info_linear_1(objstate))
        object_info = F.relu(self.graph_detection_other_info_linear_2(object_info))
        object_info = F.relu(self.graph_detection_other_info_linear_3(object_info))
        object_info = F.relu(self.graph_detection_other_info_linear_4(object_info))
        object_info = F.relu(self.graph_detection_other_info_linear_5(object_info))  # A矩阵     
        return object_info

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        objbb = model_input.objbb
        (hx, cx) = model_input.hidden
        self.gpu_id = model_input.gpu_id
        target = model_input.target_class_embedding
        target_name = model_input.target_name
        action_probs = model_input.action_probs
        objstate = self.list_from_raw_obj(objbb, target, target_name)
        x = self.gcn(objstate).unsqueeze(0).view(1, -1)
        x = torch.cat((x, action_probs), dim=1)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        image_embedding = None
        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )

    # def random_self_objects(self):
    #     random.shuffle(self.objects)

    def order_self_objects(self):
        pass

class ZeroNoGCN(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        hidden_state_sz = args.hidden_state_sz
        super(ZeroNoGCN, self).__init__()
        self.embed_action = nn.Linear(action_space, 10)

        n = 20
        feature = 5
        self.unseen_objects=[]
        lstm_input_sz = 6 + n * feature

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

        self.n = n
        # self.glove = Glove(args.glove_file)

    def list_from_raw_obj(self, objbb, target, target_name):
        glove = Glove(args.glove_file)
        objstate = torch.zeros(101, 5)
        cos = torch.nn.CosineSimilarity(dim=1)
        # glove_sim = cos(self.all_glove.detach(), target[None, :])[:, None]
        
        j = 1
        for k, v in objbb.items():
            objstate[j][0] = 1 if k == target_name else 0
            object_cos = nn.Parameter(torch.Tensor(glove.glove_embeddings[k][:]))
            object_cos.requires_grad = False
            object_cos = object_cos.to('cuda:'+str(self.gpu_id))
            objstate[j][1] = cos(object_cos.detach(), target[None, :])[:, None]
            x1 = v[0::4]
            y1 = v[1::4]
            x2 = v[2::4]
            y2 = v[3::4]
            objstate[j][2] = np.sum(x1 + x2) / len(x1 + x2) / 300
            objstate[j][3] = np.sum(y1 + y2) / len(y1 + y2) / 300
            objstate[j][4] = abs(max(x2) - min(x1)) * abs(max(y2) - min(y1)) / 300 / 300
            j = j+1
        if args.gpu_ids != -1:
            objstate = objstate.to('cuda:'+str(self.gpu_id))
        # objstate = torch.cat((objstate, glove_sim), dim=1)
        # 排序
        tmp = objstate[:,1:2]
        index = torch.argsort(tmp.squeeze(),descending=True)
        objstate = torch.index_select(objstate, 0, index=index.squeeze())
        objstate = objstate[0:self.n][:]
        return objstate

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        objbb = model_input.objbb
        (hx, cx) = model_input.hidden
        self.gpu_id = model_input.gpu_id
        target = model_input.target_class_embedding
        target_name = model_input.target_name
        action_probs = model_input.action_probs
        objstate = self.list_from_raw_obj(objbb, target, target_name)
        # x = self.gcn(objstate).unsqueeze(0).view(1, -1)
        x = objstate.unsqueeze(0).view(1, -1)
        x = torch.cat((x, action_probs), dim=1)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        image_embedding = None
        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )

    # def random_self_objects(self):
    #     random.shuffle(self.objects)

    def order_self_objects(self):
        pass