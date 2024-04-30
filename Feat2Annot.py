#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np


Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class Feat2AnnotModel(nn.Module):
    """
    Bidirectional encoder LSTM for pose feature
    Unidirectional LSTM cell decoder outputs annotation
    """

    def __init__(self, input_size, hidden_size, target_class, dropout_rate=0.2, mlp=None):
        """Init Feat2Annot Model.

        @param input_size (int): feature size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param behav_class: (int): Number of behavior annotation class
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(Feat2AnnotModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.target_class = target_class["class"]
        self.class_weight = target_class["weight"]
        # default values
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None
        self.mlp = mlp

        ## Laying out NN layers
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            device=self.device,
        )
        self.decoder = nn.LSTMCell(
            input_size=self.target_class + hidden_size,
            hidden_size=hidden_size,
            device=self.device,
        )
        # Hidden state from bidirectional LSTM is passed through a linear layer
        self.h_projection = nn.Linear(
            in_features=2 * hidden_size,
            out_features=hidden_size,
            bias=False,
            device=self.device,
        )
        # Cell from bidirectional LSTM is passed through a linear layer
        self.c_projection = nn.Linear(
            in_features=2 * hidden_size,
            out_features=hidden_size,
            bias=False,
            device=self.device,
        )
        # Attention linear layer to calculate local attention scores (multiplicative attention between hidden of encoder and hidden of decoder)
        self.att_projection = nn.Linear(
            in_features=2 * hidden_size,
            out_features=hidden_size,
            bias=False,
            device=self.device,
        )
        # Takes the combined attention output and decoder (3h) project to combined output(1h)
        self.combined_output_projection = nn.Linear(
            in_features=3 * hidden_size,
            out_features=hidden_size,
            bias=False,
            device=self.device,
        )
        # Takes mlp output and project to hidden_size
        if self.mlp is not None:
            self.mlp_to_combined_out = nn.Linear(
                in_features=self.mlp.hidden_size[-1],
                out_features=hidden_size,
                bias=True,
                device=self.device,
            )
            
        # Takes combined output project to annotation class space
        self.target_annot_projection = nn.Linear(
            in_features=hidden_size if self.mlp is None else hidden_size*2,
            out_features=self.target_class,
            bias=False,
            device=self.device,
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Take a mini-batch of source feature sequences and target annotations, compute the log-likelihood of
        target annotations under the Feat2Annot model

        @param source (np.ndarray): numpy array containing behavioral features Batch x Length x Dimension
        @param target (np.ndarray): numpy array containing behavior annotation Batch x Length x 1

        @returns scores (Tensor): a variable/tensor of shape (Batch, ) representing the
                                    log-likelihood of generating target annotation for
                                    each example in the input batch.
        """
        # Get sequence lengths
        enc_hiddens, dec_init_state = self.encode(source)
        combined_outputs = self.decode(enc_hiddens, dec_init_state, target)
        # combined_outputs B,L,H
        if self.mlp is not None:
            B,L,D = source.shape
            mlp_output = self.mlp_to_combined_out(self.mlp(source.view(-1,D))[1])
            mlp_output = mlp_output.view(B,L,-1)
            combined_outputs = torch.cat((combined_outputs,mlp_output),dim=-1)
        P = F.log_softmax(self.target_annot_projection(combined_outputs), dim=-1)
        # Batch x Length x Annot categories
        # Compute log probability of generating true target annotation
        # target tensor: B x L, P: B x L x C
        target_ground_truth_annot_log_prob = torch.gather(
            P, index=target.unsqueeze(-1), dim=-1
        ).squeeze(
            -1
        )  # (N, L)
        
        # Apply two-way exponential filter at annotation altering point
        # weight_mat = exponential_weight(target)
        # weight_mat = torch.tensor(weight_mat, dtype=torch.float32, device=self.device)
        # weight_mat = (target!=0).float()
        weight_mat = self.class_weight[target]
        
        # change_point = torch.diff(torch.concat((target[:, 0:1], target), dim=1), dim=1)
        # change_point = (change_point != 0).float()
        # # pass a cov1d to bleed all the changing point by 1 (left and right)
        # kernel = torch.tensor([[[1, 1, 1]]], dtype=torch.float32, device=self.device)
        # # dimension is D-Kernel+1+Padding
        # change_point = F.conv1d(
        #     change_point.unsqueeze(1).float(), kernel, padding=1
        # ).squeeze(0)
        # change_point = (change_point > 0).float().squeeze(1)
        # # normalize weight matrix across rows
        # weight_mat = weight_mat * change_point
        weight_mat = weight_mat / weight_mat.sum(dim=1, keepdim=True)
        target_ground_truth_annot_log_prob = (
            target_ground_truth_annot_log_prob * weight_mat
        )
        
        
        scores = target_ground_truth_annot_log_prob.sum(dim=0)
        return scores

    def encode(
        self, source: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the encoder to source feature sequences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source (Tensor): Tensor of features with shape (batch, length, num_features)
        @param source_lengths (int): sequence length (source.shape[1])
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (batch, length, h*2).
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None
        enc_hiddens, (last_hidden, last_cell) = self.encoder(source)
        # last_hidden & last_cell (2(bidirect),1,N,h)
        init_decoder_hidden = torch.cat(
            (last_hidden[0, :], last_hidden[1, :]), dim=1
        )  # (N,2h)
        init_decoder_hidden = self.h_projection(init_decoder_hidden)
        init_decoder_cell = torch.cat(
            (last_cell[0, :], last_cell[1, :]), dim=1
        )  # (N,2h)
        init_decoder_cell = self.c_projection(init_decoder_cell)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return enc_hiddens, dec_init_state

    def decode(
        self,
        enc_hiddens: torch.Tensor,
        dec_init_state: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, source_length, h*2)
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target ground truth annotation (b, source_length). target length = source length
        @returns combined_outputs (Tensor): combined output tensor  (b, source_length, h)
        """
        # Convert target to one hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.target_class)
        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state
        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device = self.device)
        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = target_one_hot  # batch x length x behave categories
        for Y_t in torch.split(Y, 1, 1):
            Y_t = torch.squeeze(Y_t, 1)
            # Concatenate o_prev to Y_t to get Ybar_t batch x (behav_categories + hidden_size)
            Ybar_t = torch.cat((Y_t, o_prev), 1)
            dec_state, o_t, e_t = self.step(
                Ybar_t=Ybar_t,
                dec_state=dec_state,
                enc_hiddens=enc_hiddens,
                enc_hiddens_proj=enc_hiddens_proj,
            )
            o_prev = o_t
            combined_outputs.append(o_t)

        combined_outputs = torch.stack(combined_outputs, 1)

        return combined_outputs

    def step(
        self,
        Ybar_t: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (N, target_class + hidden_size)
        @param dec_state (tuple(State:Tensor, Cell:Tensor)): Tuple of tensors both with shape (N, h).
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (N, source_length, 2h).
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from 2h to h. Tensor is with shape (N, source_length, h).

        @returns dec_state (tuple (State:Tensor, Cell:Tensor)): Tuple of tensors both shape (N, h).
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (N, h).
        @returns e_t (Tensor): Tensor of shape (N, source_length). Attention scores.
        """

        combined_output = None
        # Pass through decoder LSTM cell
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state  # dec_hidden (N,h), dec_cell (N,h).
        # Compute attention score using batch matrix multiplication torch.bmm
        e_t = torch.bmm(
            enc_hiddens_proj, dec_hidden.unsqueeze(-1)
        )  # enc_hiddens_proj (N, src_len, h), dec_hidden (N,h)
        # e_t (b,src_len,1)
        e_t = e_t.squeeze(-1)
        # Softmax attention scores
        alpha_t = nn.functional.softmax(e_t, dim=1)  # alpha_t (b,src_len)
        # Compute attention output using torch.bmm, weighted sum of enc_hiddens (N,src_len,2h), weights are the attention scores
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(
            1
        )  # (b,1,src_len) (b, src_len, 2h) --> (b, 2h)
        # Concatenate attention output and decoder hidden
        U_t = torch.cat((dec_hidden, a_t), dim=1)  # (b, 3h)
        V_t = self.combined_output_projection(U_t)  # (b, h)
        # Compute combined output O_t
        O_t = self.dropout(torch.tanh(V_t))
        return dec_state, O_t, e_t

    def beam_search(
        self,
        src_seq: torch.Tensor,
        beam_size: int = 5,
        max_decoding_time_step: int = None,
    ) -> List[Hypothesis]:
        """Given a single source sequence, perform beam search, yielding predicted annotation.
        @param src_sent (List[str]): a single source behavior feature seq
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        B,L,D = src_seq.shape
        if self.mlp is not None:
            _,mlp_output = self.mlp(src_seq.view((-1,D))) # BxL,D
            mlp_output = self.mlp_to_combined_out(mlp_output) # BxL, hidden_size
            mlp_output = mlp_output.view((B,L,-1))
        else:
            mlp_output = None
    
        src_encodings, dec_init_vec = self.encode(src_seq)  # (1, src_len, 2h)
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)
        
        # instead of initialize with 0, we start from the inferred class
        hypotheses = [[0]]
        lgts,_= self.mlp(src_seq[:,0,:].squeeze(1))
        a_hat = torch.argmax(lgts,dim=-1).item()
        hypotheses = [[a_hat]]
        

        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        if max_decoding_time_step is None:
            max_decoding_time_step = src_seq.shape[1]

        t = 0
        while t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            exp_src_encodings = src_encodings.expand(
                hyp_num, src_encodings.size(1), src_encodings.size(2)
            )

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(
                hyp_num,
                src_encodings_att_linear.size(1),
                src_encodings_att_linear.size(2),
            )

            y_tm1 = torch.tensor(
                [hyp[-1] for hyp in hypotheses], dtype=torch.long, device=self.device
            )  # (num_hypothe, 1)
            y_t_one_hot = F.one_hot(y_tm1, self.target_class)  # (num_hypo, behav_class)
            x = torch.cat(
                [y_t_one_hot, att_tm1], dim=-1
            )  # (num_hypo, behav_class+hidden)

            (h_t, cell_t), att_t, _ = self.step(
                x,
                h_tm1,
                exp_src_encodings,
                exp_src_encodings_att_linear,
            )
            # att_t combined output, (num_hypo, hidden size)
            # log probabilities over target words
            mlp_output_t = mlp_output[:,t-1,:].squeeze(1)
            att_t_cmbed = torch.cat((att_t,mlp_output_t.expand(att_t.shape[0],-1)),dim=-1)
            log_p_t = F.log_softmax(
                self.target_annot_projection(att_t_cmbed), dim=-1
            )  # (num_hypo, behav_class)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (
                hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t
            ).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num
            )

            prev_hyp_ids = top_cand_hyp_pos // self.target_class
            hyp_word_ids = top_cand_hyp_pos % self.target_class

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                prev_hyp_id = prev_hyp_id.item()
                hyp_annot_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                new_hyp_annot_seq = hypotheses[prev_hyp_id] + [hyp_annot_id]
                new_hypotheses.append(new_hyp_annot_seq)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                live_hyp_ids, dtype=torch.long, device=self.device
            )
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item())
            )

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """Determine which device to place the Tensors upon, CPU or GPU."""
        return self.class_weight.device

    @staticmethod
    def load(model_path: str):
        """Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = Feat2AnnotModel(**args)
        model.load_state_dict(params["state_dict"])

        return model

    def save(self, path: str):
        """Save the odel to a file.
        @param path (str): path to the model
        """
        print(f"save model parameters to {path}")

        params = {
            "args": dict(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                target_class={"class": self.target_class, "weight": self.class_weight},
                dropout_rate=self.dropout_rate,
            ),
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)


def exponential_weight(target, a=0.5):
    """
    Takes a target sequence, pass symmetric exponential filter
    """
    b, l = target.shape
    weight = np.concatenate((np.zeros((b, 1)), np.diff(target.numpy(), axis=1)), axis=1)
    # First find the ts where the label changed
    weight = (weight != 0).astype(float)
    weight_left = weight.copy()
    weight_right = weight.copy()
    for idx in range(1, weight.shape[1]):
        weight_left[:, idx] = weight_left[:, idx - 1] * a + (1 - a) * weight[:, idx]
        weight_right[:, l - idx - 1] = (
            weight_right[:, l - idx] * a + (1 - a) * weight[:, l - idx]
        )
    weight = weight_right + weight_left - (1 - a) * weight
    weight = weight / np.sum(weight, axis=1, keepdims=True)
    return weight


class Feat2AnnotFCModel(nn.Module):
    def __init__(self, input_size, hidden_size, target_class, dropout_rate=0.3, device = None):
        super(Feat2AnnotFCModel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.target_class = target_class["class"]
        self.class_weight = target_class["weight"]
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Laying out NN layers
        self.seq_layers = nn.Sequential()
        hidden_size = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        ldims = [input_size] + hidden_size
        for idx,l in enumerate(ldims):
            if idx+1 >= len(ldims):
                break
            self.seq_layers.append(nn.Linear(l, ldims[idx+1],device=self.device))
            self.seq_layers.append(nn.ReLU())
            self.seq_layers.append(nn.BatchNorm1d(ldims[idx+1]))
            self.seq_layers.append(nn.Dropout(self.dropout_rate))
            
        self.logit_layer = nn.Linear(ldims[-1], self.target_class,device=self.device)
    
    def forward(self, x):
        output = self.seq_layers(x)
        logits = self.logit_layer(output)
        return logits, output
    
    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = Feat2AnnotFCModel(**args)
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, path: str):
        print(f"save model parameters to {path}")
        params = {
            "args": dict(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                target_class={"class": self.target_class, "weight": self.class_weight},
                dropout_rate=self.dropout_rate,
                
            ),
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)

        
            
            
        
