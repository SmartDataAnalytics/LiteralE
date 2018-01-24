import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal, xavier_uniform
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

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
        xavier_normal(self.emb_e_real.weight.data)
        xavier_normal(self.emb_e_img.weight.data)
        xavier_normal(self.emb_rel_real.weight.data)
        xavier_normal(self.emb_rel_img.weight.data)

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
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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

        self.emb_num_lit = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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

class DistMultLiteral_attention(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(DistMultLiteral_attention, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)
        self.entity_specific_literal = torch.nn.Embedding(num_entities, self.n_num_lit)

        self.emb_num_lit = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.entity_specific_literal.weight.data)
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_num_weighted_literal = e1_num_lit*self.entity_specific_literal(e1).view(-1,self.n_num_lit)
        e1_emb = self.emb_num_lit(torch.cat([e1_emb, e1_num_weighted_literal], 1))        
        e2_num_weighted_literal = self.numerical_literals * self.entity_specific_literal.weight
                            
        e2_multi_emb = self.emb_num_lit(torch.cat([self.emb_e.weight, e2_num_weighted_literal], 1))

        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb*rel_emb, e2_multi_emb.t())
        pred = F.sigmoid(pred)

        return pred

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
        xavier_normal(self.emb_e_real.weight.data)
        xavier_normal(self.emb_e_img.weight.data)
        xavier_normal(self.emb_rel_real.weight.data)
        xavier_normal(self.emb_rel_img.weight.data)

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
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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


class ConvELiteralAlt(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(ConvELiteralAlt, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim)

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
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)
        xavier_normal(self.emb_num_lit.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

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

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        x = self.emb_num_lit(torch.cat([x, e1_num_lit], 1))

        e2_multi_emb = self.emb_num_lit(torch.cat([self.emb_e.weight, self.numerical_literals], 1))

        # End literals

        x = torch.mm(x, e2_multi_emb.t())
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


class DistMultLiteralText(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals):
        super(DistMultLiteralText, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        # Text literal
        self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = self.text_literals.size(1)

        self.emb_lits = torch.nn.Linear(self.emb_dim+self.n_num_lit+self.n_txt_lit, self.emb_dim)

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)
        xavier_normal(self.emb_lits.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_txt_lit = self.text_literals[e1.view(-1)]

        e1_emb = self.emb_lits(torch.cat([e1_emb, e1_num_lit, e1_txt_lit], 1))
        e1_emb = F.tanh(e1_emb)

        e2_multi_emb = self.emb_lits(torch.cat([self.emb_e.weight, self.numerical_literals, self.text_literals], 1))
        e2_multi_emb = F.tanh(e2_multi_emb)

        # End literals

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        pred = torch.mm(e1_emb*rel_emb, e2_multi_emb.t())
        pred = F.sigmoid(pred)

        return pred


class ComplexLiteralText(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals):
        super(ComplexLiteralText, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e_real = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        # Text literals
        self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = self.text_literals.size(1)

        self.emb_lit_real = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit+self.n_txt_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        self.emb_lit_img = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit+self.n_txt_lit, self.emb_dim),
            torch.nn.Tanh()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e_real.weight.data)
        xavier_normal(self.emb_e_img.weight.data)
        xavier_normal(self.emb_rel_real.weight.data)
        xavier_normal(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_emb_real = self.emb_e_real(e1).view(Config.batch_size, -1)
        rel_emb_real = self.emb_rel_real(rel).view(Config.batch_size, -1)
        e1_emb_img = self.emb_e_img(e1).view(Config.batch_size, -1)
        rel_emb_img = self.emb_rel_img(rel).view(Config.batch_size, -1)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_txt_lit = self.text_literals[e1.view(-1)]

        e1_emb_real = self.emb_lit_real(torch.cat([e1_emb_real, e1_num_lit, e1_txt_lit], 1))
        e1_emb_img = self.emb_lit_img(torch.cat([e1_emb_img, e1_num_lit, e1_txt_lit], 1))

        e2_multi_emb_real = self.emb_lit_real(torch.cat([self.emb_e_real.weight, self.numerical_literals, self.text_literals], 1))
        e2_multi_emb_img = self.emb_lit_img(torch.cat([self.emb_e_img.weight, self.numerical_literals, self.text_literals], 1))

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


class ConvELiteralText(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals):
        super(ConvELiteralText, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        # Text literal
        self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = self.text_literals.size(1)

        self.emb_lits = torch.nn.Linear(self.emb_dim+self.n_num_lit+self.n_txt_lit, self.emb_dim)

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
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_emb = self.emb_e(e1).view(Config.batch_size, -1)
        rel_emb = self.emb_rel(rel)

        # Begin literals

        e1_num_lit = self.numerical_literals[e1.view(-1)]
        e1_txt_lit = self.text_literals[e1.view(-1)]

        e1_emb = self.emb_lits(torch.cat([e1_emb, e1_num_lit, e1_txt_lit], 1))
        e2_multi_emb = self.emb_lits(torch.cat([self.emb_e.weight, self.numerical_literals, self.text_literals], 1))

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


class DistMultLiteralNN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(DistMultLiteralNN, self).__init__()

        self.emb_dim = Config.embedding_dim
        self.h_dim = 50  # Bottleneck

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = torch.nn.Sequential(
            # torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(Config.dropout),
            torch.nn.Linear(self.h_dim, self.emb_dim)
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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


class DistMultLiteralNN2(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals):
        super(DistMultLiteralNN2, self).__init__()

        self.emb_dim = Config.embedding_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim+self.n_num_lit, self.emb_dim),
            torch.nn.ReLU()
        )

        # Dropout + loss
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

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


# Add your own model here


class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = F.sigmoid(output)

        return prediction
