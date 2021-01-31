# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFN(nn.Module):
    def __init__(self, Config, NNpConfig, Gamma1Config, Gamma2Config, OutConfig):
        super(MFN, self).__init__()
        [self.d_l, self.d_a, self.d_v] = Config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = Config["h_dims"]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = Config["memsize"]
        output_dim = 1
        # output_dim = 3
        gammaInShape = total_h_dim + self.mem_dim
        final_out = total_h_dim + self.mem_dim

        h_att = Config["shapes"]
        h_nn_p = NNpConfig["shapes"]
        h_gamma1 = Gamma1Config["shapes"]
        h_gamma2 = Gamma2Config["shapes"]
        h_out = OutConfig["shapes"]

        nn_att_dropout = Config["drop"]
        nn_p_dropout = NNpConfig["drop"]
        gamma1_dropout = Gamma1Config["drop"]
        gamma2_dropout = Gamma2Config["drop"]
        out_dropout = OutConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att = nn.Sequential(nn.Linear(total_h_dim, h_att), nn.ReLU(), nn.Dropout(nn_att_dropout),
                                 nn.Linear(h_att, total_h_dim))

        self.nn_p = nn.Sequential(nn.Linear(total_h_dim, h_nn_p), nn.ReLU(),
                                  nn.Dropout(nn_p_dropout), nn.Linear(h_nn_p, self.mem_dim), nn.Tanh())

        self.nn_gmma1 = nn.Sequential(nn.Linear(gammaInShape, h_gamma1), nn.ReLU(),
                                      nn.Dropout(gamma1_dropout), nn.Linear(h_gamma1, self.mem_dim), nn.Sigmoid())
        self.nn_gmma2 = nn.Sequential(nn.Linear(gammaInShape, h_gamma2), nn.ReLU(),
                                      nn.Dropout(gamma2_dropout), nn.Linear(h_gamma2, self.mem_dim), nn.Sigmoid())

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout),
                                    nn.Linear(h_out, output_dim))

    def forward(self, x):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        # x is T*N*d
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).cuda()
        self.h_a = torch.zeros(N, self.dh_a).cuda()
        self.h_v = torch.zeros(N, self.dh_v).cuda()
        self.c_l = torch.zeros(N, self.dh_l).cuda()
        self.c_a = torch.zeros(N, self.dh_a).cuda()
        self.c_v = torch.zeros(N, self.dh_v).cuda()
        self.mem = torch.zeros(N, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(T):
            # current time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            attention = F.softmax(self.att(new_cs), dim=1)
            attended = attention * new_cs

            cHat = self.nn_p(attended)
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = self.nn_gmma1(both)
            gamma2 = self.nn_gmma2(both)
            self.mem = gamma1 * self.mem + gamma2 * cHat

            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v

            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)
            all_mems.append(self.mem)

        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        output = self.nn_out(last_hs)
        return output


class TMAN1(nn.Module):
    """
    LSTM + TMAN + MMGM
    """
    def __init__(self, Config, NNlConfig, NNaConfig, NNvConfig, NNpConfig, Gamma1Config, Gamma2Config, OutConfig):
        super(TMAN1, self).__init__()
        [self.d_l, self.d_a, self.d_v] = Config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = Config["h_dims"]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = Config["memsize"]
        self.senti_dim = Config["sentisize"]
        output_dim = 1
        # output_dim = 3
        gammaInShape = self.senti_dim + self.mem_dim
        final_out = total_h_dim + self.mem_dim

        h_att = Config["shapes"]
        h_nn_l = NNlConfig["shapes"]
        h_nn_a = NNaConfig["shapes"]
        h_nn_v = NNvConfig["shapes"]
        h_nn_p = NNpConfig["shapes"]
        h_gamma1 = Gamma1Config["shapes"]
        h_gamma2 = Gamma2Config["shapes"]
        h_out = OutConfig["shapes"]

        nn_att_dropout = Config["drop"]
        nn_l_dropout = NNlConfig["drop"]
        nn_a_dropout = NNaConfig["drop"]
        nn_v_dropout = NNvConfig["drop"]
        nn_p_dropout = NNpConfig["drop"]
        gamma1_dropout = Gamma1Config["drop"]
        gamma2_dropout = Gamma2Config["drop"]
        out_dropout = OutConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att = nn.Sequential(nn.Linear(total_h_dim, h_att), nn.ReLU(), nn.Dropout(nn_att_dropout),
                                 nn.Linear(h_att, 3))

        self.nn_l = nn.Sequential(nn.Linear(self.dh_l, h_nn_l), nn.ReLU(), nn.Dropout(nn_l_dropout),
                                  nn.Linear(h_nn_l, self.senti_dim))
        self.nn_a = nn.Sequential(nn.Linear(self.dh_a, h_nn_a), nn.ReLU(), nn.Dropout(nn_a_dropout),
                                  nn.Linear(h_nn_a, self.senti_dim))
        self.nn_v = nn.Sequential(nn.Linear(self.dh_v, h_nn_v), nn.ReLU(), nn.Dropout(nn_v_dropout),
                                  nn.Linear(h_nn_v, self.senti_dim))

        self.nn_p = nn.Sequential(nn.Linear(self.senti_dim, h_nn_p), nn.ReLU(),
                                  nn.Dropout(nn_p_dropout), nn.Linear(h_nn_p, self.mem_dim), nn.Tanh())

        self.nn_gmma1 = nn.Sequential(nn.Linear(gammaInShape, h_gamma1), nn.ReLU(),
                                      nn.Dropout(gamma1_dropout), nn.Linear(h_gamma1, self.mem_dim), nn.Sigmoid())
        self.nn_gmma2 = nn.Sequential(nn.Linear(gammaInShape, h_gamma2), nn.ReLU(),
                                      nn.Dropout(gamma2_dropout), nn.Linear(h_gamma2, self.mem_dim), nn.Sigmoid())

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout),
                                    nn.Linear(h_out, output_dim))

    def forward(self, x):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        # x is T*N*d
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).cuda()
        self.h_a = torch.zeros(N, self.dh_a).cuda()
        self.h_v = torch.zeros(N, self.dh_v).cuda()
        self.c_l = torch.zeros(N, self.dh_l).cuda()
        self.c_a = torch.zeros(N, self.dh_a).cuda()
        self.c_v = torch.zeros(N, self.dh_v).cuda()
        self.mem = torch.zeros(N, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(T):
            # current time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            attention = F.softmax(self.att(new_cs), dim=1)
            senti_l = self.nn_l(new_c_l)
            senti_a = self.nn_a(new_c_a)
            senti_v = self.nn_v(new_c_v)
            attended = torch.squeeze(torch.bmm(torch.unsqueeze(attention, dim=1), torch.stack((senti_l, senti_a, senti_v), dim=1)))

            cHat = self.nn_p(attended)
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = self.nn_gmma1(both)
            gamma2 = self.nn_gmma2(both)
            self.mem = gamma1 * self.mem + gamma2 * cHat

            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v

            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)
            all_mems.append(self.mem)

        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        output = self.nn_out(last_hs)
        return output


class LSTHM(nn.Module):
    def __init__(self, cell_size, in_size, hybrid_in_size):
        super(LSTHM, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4*self.cell_size)
        self.U = nn.Linear(cell_size, 4*self.cell_size)
        self.V = nn.Linear(hybrid_in_size, 4*self.cell_size)

    def forward(self, x, ctm, htm, ztm):
        input_affine = self.W(x)
        output_affine = self.U(htm)
        hybrid_affine = self.V(ztm)

        sums = input_affine + output_affine + hybrid_affine

        # biases are already part of W and U and V
        f_t = torch.sigmoid(sums[:, :self.cell_size])
        i_t = torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])
        o_t = torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])
        ch_t = torch.tanh(sums[:, 3*self.cell_size:])
        c_t = f_t*ctm + i_t*ch_t
        h_t = torch.tanh(c_t)*o_t

        return c_t, h_t


class MARN(nn.Module):
    def __init__(self, Config, ReduceDimConfig, MapNNConfig, OutConfig):
        super(MARN, self).__init__()
        [self.d_l, self.d_a, self.d_v] = Config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = Config["h_dims"]
        [self.l_reduce_dim, self.a_reduce_dim, self.v_reduce_dim] = ReduceDimConfig["h_dims"]
        self.total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.total_reduce_dim = self.l_reduce_dim + self.a_reduce_dim + self.v_reduce_dim
        self.num_atts = Config['num_atts']
        output_dim = 1
        # output_dim = 3
        final_out = 2 * self.total_h_dim
        h_out = OutConfig["shapes"]
        out_dropout = OutConfig["drop"]
        map_h = MapNNConfig["shapes"]
        map_dropout = MapNNConfig["drop"]

        self.lsthm_l = LSTHM(self.dh_l, self.d_l, self.total_h_dim)
        self.lsthm_a = LSTHM(self.dh_a, self.d_a, self.total_h_dim)
        self.lsthm_v = LSTHM(self.dh_v, self.d_v, self.total_h_dim)

        self.att = nn.Sequential(nn.Linear(self.total_h_dim, self.num_atts * self.total_h_dim))

        self.reduce_dim_nn_l = nn.Sequential(nn.Linear(self.num_atts * self.dh_l, self.l_reduce_dim))
        self.reduce_dim_nn_a = nn.Sequential(nn.Linear(self.num_atts * self.dh_a, self.a_reduce_dim))
        self.reduce_dim_nn_v = nn.Sequential(nn.Linear(self.num_atts * self.dh_v, self.v_reduce_dim))

        self.fc = nn.Sequential(nn.Linear(self.total_reduce_dim, map_h), nn.ReLU(), nn.Dropout(map_dropout), nn.Linear(map_h, self.total_h_dim))

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout), nn.Linear(h_out, output_dim))

    def forward(self, x):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        # x is T*N*d
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).cuda()
        self.h_a = torch.zeros(N, self.dh_a).cuda()
        self.h_v = torch.zeros(N, self.dh_v).cuda()
        self.c_l = torch.zeros(N, self.dh_l).cuda()
        self.c_a = torch.zeros(N, self.dh_a).cuda()
        self.c_v = torch.zeros(N, self.dh_v).cuda()
        self.z_t = torch.zeros(N, self.total_h_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_z_ts = []
        for i in range(T):
            # current time step
            new_c_l, new_h_l  = self.lsthm_l(x_l[i], *(self.c_l, self.h_l, self.z_t))
            new_c_a, new_h_a  = self.lsthm_a(x_a[i], *(self.c_a, self.h_a, self.z_t))
            new_c_v, new_h_v  = self.lsthm_v(x_v[i], *(self.c_v, self.h_v, self.z_t))

            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            attention = F.softmax(torch.cat(torch.chunk(self.att(new_cs), self.num_atts, dim=1), dim=0), dim=1)
            attended = attention * new_cs.repeat(self.num_atts, 1)
            reduce_l = self.reduce_dim_nn_l(torch.cat(torch.chunk(attended[:, :self.dh_l], self.num_atts, dim=0), dim=1))
            reduce_a = self.reduce_dim_nn_a(torch.cat(torch.chunk(attended[:, self.dh_l:self.dh_l+self.dh_a], self.num_atts, dim=0), dim=1))
            reduce_v = self.reduce_dim_nn_v(torch.cat(torch.chunk(attended[:, self.dh_l+self.dh_a:], self.num_atts, dim=0), dim=1))
            self.z_t = self.fc(torch.cat([reduce_l, reduce_a, reduce_v], dim=1))
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)
            all_z_ts.append(self.z_t)

        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_z_t = all_z_ts[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_z_t], dim=1)
        output = self.nn_out(last_hs)
        return output


class TMAN2(nn.Module):
    """
    LSTHM + TMAN
    """
    def __init__(self, Config, NNlConfig, NNaConfig, NNvConfig, OutConfig):
        super(TMAN2, self).__init__()
        [self.d_l, self.d_a, self.d_v] = Config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = Config["h_dims"]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.senti_dim = Config["sentisize"]
        output_dim = 1
        # output_dim = 3
        final_out = total_h_dim + self.senti_dim

        h_att = Config["shapes"]
        h_nn_l = NNlConfig["shapes"]
        h_nn_a = NNaConfig["shapes"]
        h_nn_v = NNvConfig["shapes"]
        h_out = OutConfig["shapes"]

        nn_att_dropout = Config["drop"]
        nn_l_dropout = NNlConfig["drop"]
        nn_a_dropout = NNaConfig["drop"]
        nn_v_dropout = NNvConfig["drop"]
        out_dropout = OutConfig["drop"]

        self.lsthm_l = LSTHM(self.dh_l, self.d_l, self.senti_dim)
        self.lsthm_a = LSTHM(self.dh_a, self.d_a, self.senti_dim)
        self.lsthm_v = LSTHM(self.dh_v, self.d_v, self.senti_dim)

        self.nn_l = nn.Sequential(nn.Linear(self.dh_l, h_nn_l), nn.ReLU(), nn.Dropout(nn_l_dropout),
                                  nn.Linear(h_nn_l, self.senti_dim))
        self.nn_a = nn.Sequential(nn.Linear(self.dh_a, h_nn_a), nn.ReLU(), nn.Dropout(nn_a_dropout),
                                  nn.Linear(h_nn_a, self.senti_dim))
        self.nn_v = nn.Sequential(nn.Linear(self.dh_v, h_nn_v), nn.ReLU(), nn.Dropout(nn_v_dropout),
                                  nn.Linear(h_nn_v, self.senti_dim))

        self.att = nn.Sequential(nn.Linear(total_h_dim, h_att), nn.ReLU(), nn.Dropout(nn_att_dropout), nn.Linear(h_att, 3))

        self.nn_out = nn.Sequential(nn.Linear(final_out, h_out), nn.ReLU(), nn.Dropout(out_dropout), nn.Linear(h_out, output_dim))


    def forward(self, x):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        # x is T*N*d
        N = x.shape[1]
        T = x.shape[0]
        self.h_l = torch.zeros(N, self.dh_l).cuda()
        self.h_a = torch.zeros(N, self.dh_a).cuda()
        self.h_v = torch.zeros(N, self.dh_v).cuda()
        self.c_l = torch.zeros(N, self.dh_l).cuda()
        self.c_a = torch.zeros(N, self.dh_a).cuda()
        self.c_v = torch.zeros(N, self.dh_v).cuda()
        self.senti = torch.zeros(N, self.senti_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_senti = []
        for i in range(T):
            # current time step
            new_c_l, new_h_l = self.lsthm_l(x_l[i], *(self.c_l, self.h_l, self.senti))
            new_c_a, new_h_a = self.lsthm_a(x_a[i], *(self.c_a, self.h_a, self.senti))
            new_c_v, new_h_v = self.lsthm_v(x_v[i], *(self.c_v, self.h_v, self.senti))
            # concatenate
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            senti_l = self.nn_l(new_c_l)
            senti_a = self.nn_a(new_c_a)
            senti_v = self.nn_v(new_c_v)
            attention = F.softmax(self.att(new_cs), dim=1)
            attended = torch.squeeze(torch.bmm(torch.unsqueeze(attention, dim=1), torch.stack((senti_l, senti_a, senti_v), dim=1)))
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            self.senti = attended
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)
            all_senti.append(self.senti)

        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_senti = all_senti[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_senti], dim=1)
        output = self.nn_out(last_hs)

        return output
