import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from absl import logging
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from sparsemax import row_wise_sparsemax,Sparsemax

logging.set_verbosity(logging.INFO)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False,
                 num_features=0, feature_dim=0, feature_relu_bias=2.0):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.num_features = num_features
        self.feature_dims = feature_dim
        if self.num_features == 0:
            # self.encoder = nn.Embedding(ntoken, ninp)
            self.encoder = nn.Parameter(torch.FloatTensor(ntoken, ninp))
        else:
            self.word_emb = nn.Parameter(torch.FloatTensor(ntoken, feature_dim))
            self.feature_emb = nn.Parameter(torch.FloatTensor(num_features, feature_dim))
            self.feature_relu_bias = nn.Parameter(torch.FloatTensor([feature_relu_bias]),requires_grad=False)
            self.encoder = nn.Parameter(torch.FloatTensor(num_features, ninp))
            self.word_emb_cache = None
            self.feature_emb_cache = None

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        logging.info(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert self.num_features == 0, "Its not supported to tie weights and use feature models right now."
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        if self.num_features == 0:
            logging.info('Using normal encoder model')
            self._input_layer_fn = self.normal_encoder
        else:
            logging.info('Using feature encoder model %s %s', self.num_features, self.feature_dims)
            self._input_layer_fn = self.feature_encoder

    def comp_fn(self, child, parent):
        pass

    def cache_emb(self):
        self.word_emb_cache = self.word_emb.clone().detach()
        self.feature_emb_cache = self.feature_emb.clone().detach()

    def input_layer(self):
        return self._input_layer_fn()

    def normal_encoder(self):
        return self.encoder

    def feature_encoder(self):
        # Z = F.relu(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)) - self.feature_relu_bias)
        Z = self.compute_z()
        emb = torch.matmul(Z, self.encoder)
        return emb

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.num_features > 0:
            self.word_emb.data.uniform_(-initrange, initrange)
            self.feature_emb.data.uniform_(-initrange, initrange)

    def compute_z(self):
        # Z = F.relu(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)) - self.feature_relu_bias)
        # Z = F.relu(F.softmax(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)), dim=1) - self.feature_relu_bias)
        # Z = row_wise_sparsemax.apply(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)))
        Z = F.softmax(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)), dim=1)
        # z = F.relu(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)) - self.feature_relu_bias)
        # b = Bernoulli(F.sigmoid(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0))))
        # Z = b.rsample()
        return Z

    def feature_model_sparsity_loss(self, lambda1, lambda2, avg1, avg2):
        if avg1:
            word_l2_dist = torch.mean(torch.pow(self.word_emb - self.word_emb_cache, 2))
            feat_l2_dist = torch.mean(torch.pow(self.feature_emb - self.feature_emb_cache, 2))
        else:
            word_l2_dist = torch.sum(torch.pow(self.word_emb - self.word_emb_cache, 2))
            feat_l2_dist = torch.sum(torch.pow(self.feature_emb - self.feature_emb_cache, 2))
        # z = F.relu(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0)) - self.feature_relu_bias)
        # b = Bernoulli(F.sigmoid(torch.matmul(self.word_emb, torch.transpose(self.feature_emb, 1, 0))))
        # z = b.sample()
        z = self.compute_z()
        z_sum = z.sum()
        # z_gt_0 = torch.zeros_like(z)
        z_gt_0 = torch.sign(z)
        z_gt_0_sum = z_gt_0.sum()
        if avg2:
            z_sparse = z_sum.sum()
        else:
            z_sparse = z_sum.sum(dim=1).mean()

        loss = lambda1 * word_l2_dist + lambda1 * feat_l2_dist + lambda2 * z_sparse
        logging.log_every_n(logging.INFO, 'loss %s | word %s | feat %s | z %s | z_sum %s | z > 0 %s',
                            100, loss.cpu().detach().numpy(), word_l2_dist.cpu().detach().numpy(), feat_l2_dist.cpu().detach().numpy(), z_sparse.cpu().detach().numpy(), z_sum.cpu().detach().numpy(), z_gt_0_sum.cpu().detach().numpy())
        # logging.info('z.sum() = %s, (z > 0).sum() = %s bias = %s', z.sum(), (z > 0).sum(), self.feature_relu_bias)
        return loss

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.input_layer(), input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
