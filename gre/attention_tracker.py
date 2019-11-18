""" Attention Tracker Components
"""
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Transition(nn.Module):
    def __init__(self):
        super(Transition, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    def get(self):
        raise NotImplementedError


class AttentionTracker(nn.Module):
    def __init__(self, attention, transition):
        super(AttentionTracker, self).__init__()
        self.attn_history = []
        self.trans_history = []
        self.attention = attention  # contains current attention sparse tensors of form (eg, v) -> 1
        self.transition = transition

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self, engine, *args, **kwargs):
        self.attention.initialize(*args, **kwargs)

    def get_attn(self):
        """ [Return] attn (sparse tensor): (idx, v) -> 1
        """
        return self.attention.get()
