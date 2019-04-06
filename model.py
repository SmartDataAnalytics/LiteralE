import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable

from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal_, xavier_uniform_
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import pdb

from itertools import chain

timer = CUDATimer()


class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.inp_drop(self.emb_e_real(e1)).view(Config.batch_size, -1)
        rel_embedded_real = self.inp_drop(self.emb_rel_real(rel)).view(Config.batch_size, -1)
        e1_embedded_img = self.inp_drop(self.emb_e_img(e1)).view(Config.batch_size, -1)
        rel_embedded_img = self.inp_drop(self.emb_rel_img(rel)).view(Config.batch_size, -1)

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.view(-1, Config.embedding_dim)
        rel_embedded = rel_embedded.view(-1, Config.embedding_dim)

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(Config.batch_size, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(Config.batch_size, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        #print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


"""
Literal Models
--------------
"""


class DistMultLiteral(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(DistMultLiteral, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb = self.emb_num_lit(torch.cat([e1_emb, e1_num_lit], 1))

        e2_multi_emb = self.emb_num_lit(torch.cat([self.emb_e.weight, self.numerical_literals], 1))

        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb*rel_emb, e2_multi_emb.t())
        pred = F.sigmoid(pred)

        return pred


class KBLN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, c, var):
        super(KBLN, self).__init__()

        self.num_entities = num_entities
        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        # Fixed RBF parameters
        print(c)
        print(var)
        self.c = Variable(torch.FloatTensor(c)).cuda()
        self.var = Variable(torch.FloatTensor(var)).cuda()

        # Weights for numerical, one every relation
        self.nf_weights = nn.Embedding(num_relations, self.n_num_lit)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1).view(-1, self.emb_dim)
        rel_emb = self.emb_rel(rel).view(-1, self.emb_dim)

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        score_l = torch.mm(e1_emb*rel_emb, self.emb_e.weight.t())

        """ Begin numerical literals """
        n_h = self.numerical_literals[e1.view(-1)]  # (batch_size x n_lit)
        n_t = self.numerical_literals  # (num_ents x n_lit)

        # Features (batch_size x num_ents x n_lit)
        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = self.rbf(n)
        # Weights (batch_size, 1, n_lits)
        w_nf = self.nf_weights(rel)

        # (batch_size, num_ents)
        score_n = torch.bmm(phi, w_nf.transpose(1, 2)).squeeze()
        """ End numerical literals """

        score = F.sigmoid(score_l + score_n)

        return score

    def rbf(self, n):
        """
        Apply RBF kernel parameterized by (fixed) c and var, pointwise.
        n: (batch_size, num_ents, n_lit)
        """
        return torch.exp(-(n - self.c)**2 / self.var)


class MTKGNN_DistMult(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(MTKGNN_DistMult, self).__init__()

        self.emb_dim = Config.embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_attr = torch.nn.Embedding(self.n_num_lit, self.emb_dim)

        self.attr_net_left = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_dim, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1))

        self.attr_net_right = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_dim, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1))

        self.rel_params = chain(self.emb_e.parameters(), self.emb_rel.parameters())
        self.attr_params = chain(self.emb_e.parameters(), self.emb_attr.parameters(),
            self.attr_net_left.parameters(), self.attr_net_right.parameters())

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss_rel = torch.nn.BCELoss()
        self.loss_attr = torch.nn.MSELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.view(-1, Config.embedding_dim)
        rel_embedded = rel_embedded.view(-1, Config.embedding_dim)

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred

    def forward_attr(self, e, mode='left'):
        assert mode == 'left' or mode == 'right'

        e_emb = self.emb_e(e.view(-1))

        # Sample one numerical literal for each entity
        e_attr = self.numerical_literals[e.view(-1)]
        m = len(e_attr)
        idxs = torch.randint(self.n_num_lit, size=(m,)).cuda()
        attr_emb = self.emb_attr(idxs)

        inputs = torch.cat([e_emb, attr_emb], dim=1)
        pred = self.attr_net_left(inputs) if mode == 'left' else self.attr_net_right(inputs)
        target = e_attr[range(m), idxs]

        return pred, target

    def forward_AST(self):
        m = Config.batch_size

        idxs_attr = torch.randint(self.n_num_lit, size=(m,)).cuda()
        idxs_ent = torch.randint(self.num_entities, size=(m,)).cuda()

        attr_emb = self.emb_attr(idxs_attr)
        ent_emb = self.emb_e(idxs_ent)

        inputs = torch.cat([ent_emb, attr_emb], dim=1)
        pred_left = self.attr_net_left(inputs)
        pred_right = self.attr_net_right(inputs)
        target = self.numerical_literals[idxs_ent][range(m), idxs_attr]

        return pred_left, pred_right, target


class ComplexLiteral(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ComplexLiteral, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e_real = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit_real = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        self.emb_num_lit_img = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_emb_real = self.emb_e_real(e1).view(Config.batch_size, -1)
        rel_emb_real = self.emb_rel_real(rel).view(Config.batch_size, -1)
        e1_emb_img = self.emb_e_img(e1).view(Config.batch_size, -1)
        rel_emb_img = self.emb_rel_img(rel).view(Config.batch_size, -1)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb_real = self.emb_num_lit_real(torch.cat([e1_emb_real, e1_num_lit], 1))
        e1_emb_img = self.emb_num_lit_img(torch.cat([e1_emb_img, e1_num_lit], 1))

        e2_multi_emb_real = self.emb_num_lit_real(torch.cat([self.emb_e_real.weight, self.numerical_literals], 1))
        e2_multi_emb_img = self.emb_num_lit_img(torch.cat([self.emb_e_img.weight, self.numerical_literals], 1))

        # End literals

        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)

        realrealreal = torch.mm(e1_emb_real*rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real*rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img*rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img*rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class ConvELiteral(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ConvELiteral, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, self.emb_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1).view(Config.batch_size, -1)
        rel_emb = self.emb_rel(rel)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb = self.emb_num_lit(torch.cat([e1_emb, e1_num_lit], 1))

        e2_multi_emb = self.emb_num_lit(torch.cat([self.emb_e.weight, self.numerical_literals], 1))

        # End literals

        e1_emb = e1_emb.view(Config.batch_size, 1, 10, self.emb_dim//10)
        rel_emb = rel_emb.view(Config.batch_size, 1, 10, self.emb_dim//10)

        stacked_inputs = torch.cat([e1_emb, rel_emb], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e2_multi_emb.t())
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 # gate_activation=nn.functional.softmax):
                 gate_activation=nn.functional.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class DistMultLiteral_gate(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(DistMultLiteral_gate, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)


        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(self.emb_e.weight, self.numerical_literals)

        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb*rel_emb, e2_multi_emb.t())
        pred = F.sigmoid(pred)

        return pred


class ComplexLiteral_gate(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ComplexLiteral_gate, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e_real = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit_real = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)
        self.emb_num_lit_img = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_emb_real = self.emb_e_real(e1).view(Config.batch_size, -1)
        rel_emb_real = self.emb_rel_real(rel).view(Config.batch_size, -1)
        e1_emb_img = self.emb_e_img(e1).view(Config.batch_size, -1)
        rel_emb_img = self.emb_rel_img(rel).view(Config.batch_size, -1)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_emb_real = self.emb_num_lit_real(e1_emb_real, e1_num_lit)
        e1_emb_img = self.emb_num_lit_img(e1_emb_img, e1_num_lit)

        e2_multi_emb_real = self.emb_num_lit_real(self.emb_e_real.weight, self.numerical_literals)
        e2_multi_emb_img = self.emb_num_lit_img(self.emb_e_img.weight, self.numerical_literals)

        # End literals

        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)

        realrealreal = torch.mm(e1_emb_real*rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real*rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img*rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img*rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class ConvELiteral_gate(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ConvELiteral_gate, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, self.emb_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1).view(Config.batch_size, -1)
        rel_emb = self.emb_rel(rel)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]

        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(self.emb_e.weight, self.numerical_literals)

        # End literals

        e1_emb = e1_emb.view(Config.batch_size, 1, 10, self.emb_dim//10)
        rel_emb = rel_emb.view(Config.batch_size, 1, 10, self.emb_dim//10)

        stacked_inputs = torch.cat([e1_emb, rel_emb], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e2_multi_emb.t())
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


"""
TEXT LITERALS
-----------------------------------
"""

class GateMulti(nn.Module):

    def __init__(self, emb_size, num_lit_size, txt_lit_size, gate_activation=nn.functional.sigmoid):
        super(GateMulti, self).__init__()

        self.emb_size = emb_size
        self.num_lit_size = num_lit_size
        self.txt_lit_size = txt_lit_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(emb_size+num_lit_size+txt_lit_size, emb_size)

        self.gate_ent = nn.Linear(emb_size, emb_size, bias=False)
        self.gate_num_lit = nn.Linear(num_lit_size, emb_size, bias=False)
        self.gate_txt_lit = nn.Linear(txt_lit_size, emb_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(emb_size))

    def forward(self, x_ent, x_lit_num, x_lit_txt):
        x = torch.cat([x_ent, x_lit_num, x_lit_txt], 1)
        g_embedded = F.tanh(self.g(x))
        gate = self.gate_activation(self.gate_ent(x_ent) + self.gate_num_lit(x_lit_num) + self.gate_txt_lit(x_lit_txt) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class DistMultLiteral_gate_text(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals):
        super(DistMultLiteral_gate_text, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Num. Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        # Txt. Literal
        # num_ent x n_txt_lit
        self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = self.text_literals.size(1)

        # LiteralE's g
        self.emb_lit = GateMulti(self.emb_dim, self.n_num_lit, self.n_txt_lit)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals
        # --------------
        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_txt_lit = self.text_literals[e1.view(-1)]
        e1_emb = self.emb_lit(e1_emb, e1_num_lit, e1_txt_lit)
        e2_multi_emb = self.emb_lit(self.emb_e.weight, self.numerical_literals, self.text_literals)
        # --------------
        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb*rel_emb, e2_multi_emb.t())
        pred = F.sigmoid(pred)

        return pred
