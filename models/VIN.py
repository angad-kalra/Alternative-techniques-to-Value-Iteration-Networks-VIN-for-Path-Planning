import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# VIN planner module
class Planner(nn.Module):

    def __init__(self, orient, actions, args):
        super(Planner, self).__init__()

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.orient = orient
        self.actions = actions
        self.q = nn.Conv2d(in_channels=orient,out_channels=self.l_q * orient,kernel_size=(self.f, self.f),stride=1,padding=int((self.f - 1.0) / 2),bias=False)
        self.r = nn.Conv2d(in_channels=self.l_h,out_channels=orient,kernel_size=(1, 1),stride=1,padding=0,bias=False)
        self.policy = nn.Conv2d(in_channels=self.l_q * orient,out_channels=actions * orient,kernel_size=(1, 1),stride=1,padding=0,bias=False)    
        self.h = nn.Conv2d(in_channels=(orient + 1),out_channels=self.l_h,kernel_size=(3, 3),stride=1,padding=1,bias=True)
        self.w = Parameter(torch.zeros(self.l_q * orient, orient, self.f,self.f),requires_grad=True)
        self.sm = nn.Softmax2d()

    def forward(self, design, goal):
        m_s = design.size()[-1]
        X = torch.cat([design, goal], 1)

        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        q = q.view(-1, self.orient, self.l_q, m_s, m_s)
        v, _ = torch.max(q, dim=2, keepdim=True)
        v = v.view(-1, self.orient, m_s, m_s)
        for _ in range(0, self.k - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=int((self.f - 1.0) / 2))
            q = q.view(-1, self.orient, self.l_q, m_s, m_s)
            v, _ = torch.max(q, dim=2)
            v = v.view(-1, self.orient, m_s, m_s)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=int((self.f - 1.0) / 2))

        logits = self.policy(q)

        # Normalize over actions
        logits = logits.view(-1, self.actions, m_s, m_s)
        probs = self.sm(logits)

        # Reshape to output dimensions
        logits = logits.view(-1, self.orient, self.actions, m_s,
                             m_s)
        probs = probs.view(-1, self.orient, self.actions, m_s,
                           m_s)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        return logits, probs, v, r
