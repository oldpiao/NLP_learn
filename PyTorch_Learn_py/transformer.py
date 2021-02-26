# 1.1 加载包
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


# 1.2 Linear & Softmax
class Generator(nn.Module):
    """线性层与softmax层，用于将模型预测结果转换成词向量的模型"""

    def __init__(self, d_model, vocab):
        """
        :param d_model: 模型一维长度，一般为512，方便多头计算
        :param vocab: 词典中的词数量
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        max(a*x+b),将Transformer的输出转换成概率词出现的概率
        :param x: decoder生成的Transformer的输出，size(d_model, N)
        :return:
        """
        return F.log_softmax(self.proj(x), dim=-1)


# 1.3 编码器与解码器结构
class EncoderDecoder(nn.Module):
    """一个标准的编码器-解码器架构。这是很多其他模型的基础。"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder: 编码器模型
        :param decoder: 解码器模型
        :param src_embed: 输入端嵌入层模型
        :param tgt_embed: 输出端嵌入层模型
        :param generator: Generator, 线性层与softmax层
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src: 句子向量，输入文本
        :param tgt: 句子向量，输出文本
        :param src_mask: 输入文本的掩码
        :param tgt_mask: 输出文本的掩码
        :return:
        """
        return self.decode(self.encode(src, src_mask),
                           src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        :param memory: 当前Transformer模型中即为self.encode的输出，seq2seq中可能还会再加入注意力模型等
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# 1.4 标准化与子层连接
class LayerNorm(nn.Module):
    """层标准化，add & Norm
    公式：a * (x-mean_x) / (std_x + eps) + b
    引入超参数a和b，向量长度与特征数量相同
    """
    def __init__(self, features, eps=1e-6):
        """
        :param features: 特征数量
        :param eps: 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        :param x: 嵌入后的句子矩阵，三个维度，逐个句子>逐个词>嵌入后的词矩阵
        :return:
        """
        # 由于是对嵌入后的句子矩阵进行计算，因此实际是对句子矩阵中的每个词矩阵计算，
        # 因此要选择句子对应的维度后再计算
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """在层规范之后的残差连接。
    注意，为了代码的简单性，规范是第一个，而不是最后一个。
    包含标准化模型
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """将剩余连接应用到相同大小的任何子层。"""
        return x + self.dropout(sublayer(self.norm(x)))


# 1.5 编码器
def clones(module, N):
    """模型克隆"""
    # 产生N个相同的层，N=6
    # ModuleList 可以像常规Python列表一样编制索引，包含的模块已正确注册
    # copy.copy 浅拷贝 只拷贝父对象，不会拷贝对象的内部的子对象
    # copy.deepcopy 深拷贝 拷贝对象及其子对象
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """编码器模型，内部包含多层编码器"""
    def __init__(self, layer, N):
        """
        :param layer: 编码器
        :param N: 编码器层数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 归一化层 LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
        # normalized_shape 输入尺寸  [∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]
        # eps-为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5
        # elementwise_affine 布尔值，当设为true，给该层添加可学习的仿射变换参数
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """编码器的一层模型，执行顺序为：多头注意力模型>子层连接>前向反馈>子层连接"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 1.6 解码器
class Decoder(nn.Module):
    """解码器，解码器中forward方法需要传入Decoder的输出"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: 预测结果
        :param memory: Encoder输出
        :param src_mask: Encoder mask
        :param tgt_mask: 预测结果 mask
        :return:
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """解码器的一层模型，
    执行顺序为：自注意力>子层连接>Encoder-Decoder注意力>子层连接>前向反馈>子层连接"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """连续掩码，一个上半区为False,下半区为True的维度为(size,size)的矩阵"""
    attn_shape = (1, size, size)
    mask = np.triu(torch.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


# 1.7 注意力算法
def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None, dropout=None):
    """注意力算法: softmax(Q * K_t / sqrt(d_k)) * V
    K_t是K的转置；d_k是Q的size(-1);
    :param query: 注意力矩阵，三维矩阵
    :param key: 三维矩阵
    :param value: 三维矩阵
    :param mask: 掩码，在softmax前使用
    :param dropout: 在softmax后，乘value前使用，随机去除一些值
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力模型
    执行顺序：Q/K/V>Linear/Linear/Linear>attention>concat>Linear
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 同样的掩码适用于所有的h头
            # 添加一个维度
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # QKV分别进入线性层，并改变维度
        # zip会根据少的数组确定结果长度
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 添加到注意力算法中计算
        x, self.attn = attention(query, key, value, mask)
        # 维度合并（concat）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 1.8 注意力算法在模型中的应用
# 1.9 Position-wise前馈网络
class PositionwiseFeedForward(nn.Module):
    """Positionwise前馈神经网络
    FFN = max(0, x*w_1+b_1)*w_2 + b_2
    执行过程：linear>relu>dropout>linear
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 1.10 Embedding和Softmax
class Embeddings(nn.Module):
    """
    使用预学习的Embedding将输入Token序列和输出Token序列
    转化为d_model维向量
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 1.11 位置编码
class PositionalEncoding(nn.Module):
    """位置编码
    PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})
    PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model:
        :param dropout:
        :param max_len: 默认为5000，准备了足够多的位置编码供模型使用
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # positional_encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# 1.12 完整模型
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """连接完整模型并设置超参的函数
    :param src_vocab: 输入词的词典大小（词数量）
    :param tgt_vocab: 目标词的词典大小（词数量）
    :param N: 模型层数，默认6层，可以根据计算能力与需求修改
    :param d_model: 模型嵌入时向量长度，一般设置为512，方便存储和计算
    :param d_ff: ff为Feed Forward，该参数即为正向传播的向量长度
    :param h: 多头注意力的头的数量，一般为8
    :param dropout: nn.Dropout的参数，默认是0.1，随机清理10%的数据
    :return:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)  # 教程中未设置dropout参数
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout=dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # 这一点在代码中很重要。
    # 使用Glorot / fan_avg初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)  # 旧
            nn.init.xavier_uniform_(p)  # 新
    return model


# 2 模型训练
# 2.1 批和掩码
class Batch:
    """用于在训练过程中保存带掩码的一批数据的对象"""
    def __init__(self, src, trg=None, pad=0):
        """
        :param src: 输入的句子向量
        :param trg: 对应的输出
        :param pad: 用于将句子填充到最长句的句长的填充物
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]  # 去除结束符
            self.trg_y = trg[:, 1:]  # 去除起始符
            # 掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 句子长度
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建一个掩码来隐藏填充中和未填充的单词
        由于句子的长度不同，短句需要用pad填充，在为其创建mask时需要将填充部分全部排除在外，
        :param tgt: 目标向量
        :param pad: 填充物，非句子内容，用于补全句子长度的参数
        :return:
        """
        # 将目标向量转换成bool向量，表示当前位置是否有词，并添加维度
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 再与创建的mask做与操作，创建出正确的mask
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# 2.2 训练循环
def run_epoch(data_iter, model, loss_compute):
    """
    标准的训练和跟踪功能
    :param data_iter: 待处理数据，[Batch(), Batch(), ...]
    :param model: 用于训练的模型
    :param loss_compute: 损失函数模型
    :return:
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# 2.3 训练数据和批处理
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """继续扩充batch,并计算令牌+填充的总数。"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_src_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# 2.5 优化器
class NoamOpt:
    """Optim wrapper that implements rate.
    创建一个智能调整学习速率的优化器，
    在前期模型预热阶段速率越来越大，后面逐渐减小
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        :param model_size: 前面模型中的d_model参数，模型参数维度
        :param factor: 系数，学习速率系数
        :param warmup: 预热
        :param optimizer: 优化器
        """
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            # 修改学习速率
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """学习速率（lr）"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step**(-0.5), step * self.warmup**(-1.5)))


def get_std_opt(model):
    opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.8), eps=1e-9)
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, optimizer=opt)


# 2.6 正则化
# 2.6.1 标签平滑
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # size_average=False 改为 reduction="sum"
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# 3 第一个例子
# 3.1 数据生成
def data_gen(V, batch, nbatches):
    """
    为src-tgt复制任务生成随机数据
    :param V: 生成的数据最大值，相当于词典大小
    :param batch: 批次大小，一批数据中有多少句话
    :param nbatches: 数据量
    :return:
    """
    for i in range(nbatches):
        # 从1开始，因为0是起始符
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).long()
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


# 3.2 损失计算
class SimpleLossCompute(object):
    """一个简单的损失计算和训练函数"""
    def __init__(self, generator, criterion, opt=None):
        """
        :param generator: Generator, 线性层与softmax层
        :param criterion: 损失函数
        :param opt: 优化器
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class MultiGPULossCompute(object):
    """多GPU损失函数"""
    def __init__(self, generator, criterion, devices: list, opt=None, chunk_size=5):
        """
        :param generator: 待并行计算的模型的输出层，model.generator
        :param criterion: 损失函数
        :param devices: GPU计算设备，list，例：[0,1,2]
        :param opt: 优化器
        :param chunk_size: 块大小
        """
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # 将数据拆分成更小的块，发配到不同的GPU算计算
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # 预测分布
            out_column = [[Variable(o[:, i:i+chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算损失
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i+chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # sum and normalize loss
            # 在第一个GPU上汇总数据，并求和、标准化loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0]/normalize
            total += l.data[0]

            # 输出transformer反向传播loss
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.colone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])

            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


# 3.3 贪心解码
def greedy_decode(model: EncoderDecoder, src, src_mask, max_len, start_symbol):
    """ 贪心解码
    执行流程：
        用起始符构建结果向量初始状态；
        src通过encode模型，生成编码器结果，该结果会一直在解码器中使用；
        循环解码：（循环次数为最大句子长度-1，因为起始符占用一位）
            结果向量通过decode模型（会用到encode的结果）；
            输出结果通过generator，变成各个词的概率值；
            使用贪心算法获取最佳结果，将结果最佳到结果向量中；
        输出结果向量。
    :param model: 模型
    :param src: 输入数据
    :param src_mask: 输入数据掩码
    :param max_len: 最大长度
    :param start_symbol: 起始符
    :return:
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                     Variable(ys),
                     Variable(subsequent_mask(ys.size(1)).type_as(src_mask)))
        # 最后一位是结束符
        prob = model.generator(out[:, -1])
        # 贪心算法，仅看中每个位置上权重最大的值
        # 此外还可以取前N个权重较大的值，全部带入后续训练，
        # 在后续训练中也同样保留N个权重最大的，最终选择选出一个最优解
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        # 一个词一个词预测的，预测结果添加到输出中，再放入模型继续预测下一个词。
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
    return ys


