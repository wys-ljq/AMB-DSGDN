# Modified model.py with Differential Attention GCN

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import f1_score

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context, atten_score = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out), atten_score


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b, mask, speaker_emb=None):
        # 将positon、model_feature信息相加
        if speaker_emb != None:
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
        for i in range(self.layers):
            x_b, atten_score = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        return x_b, atten_score

class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

# torch.sigmoid()更适合快速计算;nn.Sigmoid()更适合构建可训练神经网络
# nn.Sigmoid()是一个nn.Module,可以作为神经网络模块使用,具有可学习的参数,可以通过反向传播训练。torch.sigmoid()是一个固定的数学函数。
class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DiffGraphAttentionLayer(nn.Module):
    """
    Enhanced Differential Attention GCN layer adapted from the provided differential attention code.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation=True, num_relation=-1,
                 relation_dim=10, depth=1):
        super(DiffGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.alpha = alpha
        self.concat = concat
        self.relation = relation
        self.internal_dim = out_features // 2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Projections for pos and neg
        self.a_left_pos = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_right_pos = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_left_neg = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_right_neg = nn.Linear(self.internal_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.a_left_pos.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_right_pos.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_left_neg.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_right_neg.weight, gain=1.414)

        if self.relation:
            self.relation_embedding = nn.Embedding(6, relation_dim)
            self.a_rel_pos = nn.Linear(relation_dim, 1, bias=False)
            self.a_rel_neg = nn.Linear(relation_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.a_rel_pos.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a_rel_neg.weight, gain=1.414)

        # Lambda parameters
        self.lambda_left1 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_right1 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_left2 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_right2 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.lambda_init = lambda_init_fn(depth)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj):
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W)  # (B, N, D_out)
        Wh = Wh.view(-1, Wh.size(1), 2, self.internal_dim)  # (B, N, 2, internal_dim)
        Wh_pos = Wh[:, :, 0, :]
        Wh_neg = Wh[:, :, 1, :]

        # Compute e_pos and e_neg
        left_proj_pos = self.a_left_pos(Wh_pos)  # (B, N, 1)
        right_proj_pos = self.a_right_pos(Wh_pos)  # (B, N, 1)
        e_pos = left_proj_pos + right_proj_pos.transpose(1, 2)  # (B, N, N)

        left_proj_neg = self.a_left_neg(Wh_neg)  # (B, N, 1)
        right_proj_neg = self.a_right_neg(Wh_neg)  # (B, N, 1)
        e_neg = left_proj_neg + right_proj_neg.transpose(1, 2)  # (B, N, N)

        if self.relation:
            long_adj = adj.clone().type(torch.long).to(h.device)
            relation_emb = self.relation_embedding(long_adj)  # (B, N, N, relation_dim)
            rel_proj_pos = self.a_rel_pos(relation_emb).squeeze(-1)  # (B, N, N)
            rel_proj_neg = self.a_rel_neg(relation_emb).squeeze(-1)  # (B, N, N)
            e_pos = e_pos + rel_proj_pos
            e_neg = e_neg + rel_proj_neg

        e_pos = self.leakyrelu(e_pos)  # (B, N, N)
        e_neg = self.leakyrelu(e_neg)  # (B, N, N)

        # Softmax with masking if relation
        if self.relation:
            zero_vec = -9e15 * torch.ones_like(e_pos)
            e_pos_masked = torch.where(adj > 0, e_pos, zero_vec)
            e_neg_masked = torch.where(adj > 0, e_neg, zero_vec)
            attention_pos = F.softmax(e_pos_masked, dim=2)
            attention_neg = F.softmax(e_neg_masked, dim=2)
        else:
            attention_pos = F.softmax(e_pos, dim=2)
            attention_neg = F.softmax(e_neg, dim=2)

        # Lambda computation
        lambda_1 = torch.exp(torch.sum(self.lambda_left1 * self.lambda_right1, dim=-1).float()).type_as(Wh)
        lambda_2 = torch.exp(torch.sum(self.lambda_left2 * self.lambda_right2, dim=-1).float()).type_as(Wh)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention
        attention = attention_pos - lambda_full * attention_neg

        attention = F.dropout(attention, self.dropout, training=self.training)

        # V is concatenated pos and neg
        Wh_v = torch.cat((Wh_pos, Wh_neg), dim=-1)  # (B, N, out_features)
        h_prime = torch.matmul(attention, Wh_v)  # (B, N, out_features)

        h_prime = self.layer_norm(h_prime)
        h_prime = h_prime * (1 - self.lambda_init)

        if self.concat:
            return F.gelu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DiffRGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=2, num_relation=-1, depth=1):
        """Dense version of GAT."""
        super(DiffRGCN, self).__init__()
        self.dropout = dropout
        self.attentions = [DiffGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation=True,
                                               num_relation=num_relation, depth=depth) for _ in range(nheads)]  # 多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = DiffGraphAttentionLayer(nfeat * nheads, nhid, dropout=dropout, alpha=alpha, concat=True,
                                           relation=True, num_relation=num_relation, depth=depth)  # 恢复到正常维度

        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # (B,N,num_head*N_out)
        attened_outputs = []
        attention_weights = []
        for att_module in self.attentions:
            # 计算注意力模块输出
            att_out, att_w = att_module(x, adj)
            # Graphplt(att_w)
            # 添加到输出列表
            attened_outputs.append(att_out)
            attention_weights.append(att_w)
            # 沿最后一个维度拼接
        x = torch.cat(attened_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        att_out, att_w = self.out_att(x, adj)
        attention_weights.append(att_w)
        x = F.gelu(att_out)  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x, attention_weights

def Graphplt(Attention):
    Attention = Attention[-1]
    # attention = F.softmax(attention, dim=2)
    attention = Attention.cpu().detach().numpy()
    num = len(attention)
    n = math.ceil(math.sqrt(num))
    m = math.ceil(num / n)
    fig = plt.figure(figsize=(20 * n, 20 * m), dpi=75)
    for i in range(num):
        axs = fig.add_subplot(n, m, i+1)
        sns.heatmap(attention[i], cmap='coolwarm', annot=True, fmt='.2f', ax=axs)
    plt.tight_layout()
    plt.show()

class custom_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,theta):
        ctx.save_for_backward(input,theta)
        return input/(1-theta.item())

    @staticmethod
    def backward(ctx,grad_output):
        input,theta=ctx.saved_tensors
        input_grad=1/(1-theta.item())*grad_output.clone()

        return input_grad,None

class Modality_drop(nn.Module):

    def __init__(self, dim_list, p_exe=0.7, device='cuda'):
        super().__init__()
        self.dim_list=dim_list
        self.p_exe=p_exe
        self.device=device

    def execute_drop(self, fead_list, q):
        num_mod = len(fead_list)
        B = fead_list[0].shape[0]
        L = fead_list[0].shape[1]
        D = fead_list[0].shape[2]
        exe_drop = torch.rand(1, device=self.device) >= 1-self.p_exe  # 用 torch.rand 替换 np.random
        if not exe_drop:
            return fead_list, torch.ones([B],dtype=torch.int32,device=self.device)

        d_sum = sum(self.dim_list)
        q_sum = sum(self.dim_list * q)
        theta = q_sum / d_sum

        mask = torch.distributions.Bernoulli(1-q).sample([B,1]).permute(2,1,0).contiguous().reshape(num_mod,B,-1).to(device=self.device)  # [num_mod, B, 1]
        concat_list = torch.stack(fead_list, dim=0)  # [num_mod, B, L, D]
        concat_list = torch.mul(concat_list, mask.unsqueeze(2))  # broadcast mask to [num_mod, B, 1, 1]
        concat_list = custom_autograd.apply(concat_list, theta)
        mask = torch.transpose(mask, 0, 1).squeeze(-1)  # [B, num_mod]
        update_flag = torch.sum(mask, dim=1) > 0
        broadcast_update = update_flag.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, B, 1, 1]
        cleaned_concat = torch.masked_select(concat_list, broadcast_update).reshape(num_mod, -1, L, D)
        cleaned_fea = torch.chunk(cleaned_concat, num_mod, dim=0)
        cleaned_fea = [_.squeeze(0) for _ in cleaned_fea]  # list of [valid_B, L, D]
        return cleaned_fea, update_flag

def calcu_q(performances, q_base, fix_lambda):
    num_mod = len(performances)
    q = torch.zeros(num_mod, device=performances.device)
    relu = nn.ReLU(inplace=True)
    softmax = nn.Softmax(dim=0)  # 新增Softmax模块
    epsilon = 1e-5  
    for i in range(num_mod):
        ratios = []
        for j in range(num_mod):
            if i == j:
                continue
            if performances[j] < epsilon: 
                continue
            ratio = performances[i] / (performances[j] + epsilon) - 1  # 计算原始差异比率，添加 epsilon
            ratios.append(ratio)
        if ratios:
            ratios_tensor = torch.tensor(ratios, device=performances.device)
            norm_ratios = softmax(relu(ratios_tensor)) 
            avg_ratio = torch.sum(norm_ratios * ratios_tensor) / len(ratios)
            if avg_ratio > 0:
                q[i] = q_base * (1 + fix_lambda * avg_ratio)
    q = torch.clip(q, 0.0, 1.0)
    return q

class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, use_adam_drop=True,
                 q_base=0.3, lam=0.9, p_exe=0.5):
        super(Transformer_Based_Model, self).__init__()
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dataset = dataset
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        self.textf_input = nn.Linear(D_text, hidden_dim)
        self.acouf_input = nn.Linear(D_audio, hidden_dim)
        self.visuf_input = nn.Linear(D_visual, hidden_dim)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.agate = EnhancedFilterModule(hidden_dim)
        self.vgate = EnhancedFilterModule(hidden_dim)

        # Inter-Speaker
        self.gatTer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatT = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)

        # Add GAT for audio and visual
        self.gatAer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatA = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatVer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatV = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )

        # 新增开关参数
        self.use_adam_drop = use_adam_drop
        self.q_base = q_base
        self.lam = lam
        self.p_exe = p_exe
        self.d = [hidden_dim, hidden_dim, hidden_dim]
        self.modality_drop = Modality_drop(dim_list=torch.tensor(self.d), p_exe=self.p_exe)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, Self_semantic_adj, Cross_semantic_adj, Semantic_adj, label=None, warm_up=1):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(spk_idx.device)
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(spk_idx.device)

        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 0, 2))
        textf, Cattention_weights = self.gatTer(textf, Cross_semantic_adj)
        textf, Sattention_weights = self.gatT(textf, Self_semantic_adj)
        B = textf.shape[0]

        sub_log_prog = []
        if visuf is not None and acouf is not None:
            acouf = self.acouf_input(acouf.permute(1, 0, 2))
            acouf, attention_weights = self.a_a(acouf, u_mask, spk_embeddings)
            acouf, Cattention_weights_a = self.gatAer(acouf, Cross_semantic_adj)
            acouf, Sattention_weights_a = self.gatA(acouf, Self_semantic_adj)
            acouf = self.agate(acouf)
            visuf = self.visuf_input(visuf.permute(1, 0, 2))
            visuf, attention_weights = self.v_v(visuf, u_mask, spk_embeddings)
            visuf, Cattention_weights_v = self.gatVer(visuf, Cross_semantic_adj)
            visuf, Sattention_weights_v = self.gatV(visuf, Self_semantic_adj)
            visuf = self.vgate(visuf)
            fead_list = [textf, visuf, acouf]
        else:
            fead_list = [textf]

        num_mod = len(fead_list)
        device = textf.device
        update_flag = torch.ones([B], dtype=torch.int32, device=device)
        if self.use_adam_drop and label is not None and warm_up == 0:
            # compute perfs
            t = self.t_output_layer(textf)
            perf_t = self._get_performance(t, label, u_mask)
            if visuf is not None and acouf is not None:
                a = self.a_output_layer(acouf)
                v = self.v_output_layer(visuf)
                perf_a = self._get_performance(a, label, u_mask)
                perf_v = self._get_performance(v, label, u_mask)
                performances = torch.tensor([perf_t, perf_v, perf_a], device=device)
            else:
                performances = torch.tensor([perf_t], device=device)
            q = calcu_q(performances, self.q_base, self.lam)
            dim_list = torch.tensor([textf.shape[-1]] * num_mod, device=device)
            self.modality_drop.dim_list = dim_list
            cleaned_fea, update_flag = self.modality_drop.execute_drop(fead_list, q)
            textf = cleaned_fea[0]
            if visuf is not None and acouf is not None:
                visuf = cleaned_fea[1]
                acouf = cleaned_fea[2]

        t = self.t_output_layer(textf)
        if visuf is not None and acouf is not None:
            a = self.a_output_layer(acouf)
            v = self.v_output_layer(visuf)
            all_final_out = t+a+v
            sub_log_prog.append(F.log_softmax(t, dim=-1))
            sub_log_prog.append(F.log_softmax(a, dim=-1))
            sub_log_prog.append(F.log_softmax(v, dim=-1))
        else:
            all_final_out = t
            sub_log_prog.append(F.log_softmax(t, dim=-1))
        all_log_prob = F.log_softmax(all_final_out, dim=-1)
        all_prob = F.softmax(all_final_out, dim=-1)
        return sub_log_prog, all_log_prob, all_prob, all_final_out, update_flag

    def _get_performance(self, logit, label, umask):
        if logit is None:
            return 0.0
        flat_logit = logit.view(-1, self.n_classes)
        flat_label = label.view(-1)
        flat_umask = umask.view(-1) == 1
        valid_preds = torch.argmax(flat_logit[flat_umask], dim=-1)
        valid_labels = flat_label[flat_umask]
        if len(valid_labels) == 0:
            return 0.0
        f1 = f1_score(valid_labels.cpu().numpy(), valid_preds.cpu().numpy(), average='weighted')
        return f1  # 返回F1