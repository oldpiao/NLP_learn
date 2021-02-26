# 德语转英语的翻译器
# 4.1 数据加载
from transformer import *
from torchtext import data, datasets


if True:
    import spacy

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    print(spacy_de.__dict__)
    print(spacy_en.__dict__)

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = "<s>"  # 句子起始符
    EOS_WORD = "</s>"  # 句子结束符
    BLACK_WORD = "<blank>"  # 句子填充符（将句子补全成相同长度）
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLACK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLACK_WORD)

    MAX_LEN = 100  # 最大句子长度

    # 注意此处加载了de-en.tgz数据，但需要按指定格式配置路径
    # 由于当前网络未翻墙，因此是离线下载的数据
    # 数据目录结构：.data/iwslt/de-en
    # root参数配置时配置：.data
    train, val, test = datasets.IWSLT.splits(
        exts=(".de", ".en"), fields=(SRC, TGT),
        root="../../data/英语-德语翻译/.data",
        filter_pred=lambda x: len(vars(x)["src"]) < MAX_LEN and
                              len(vars(x)["trg"]) < MAX_LEN)
    MIN_FREQ = 2  # 最短词长，比这个短的不再作为词使用
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


# 4.2 迭代器
class MyIterator(data.Iterator):
    def create_batches(self):
        """创建数据批次"""
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """修正torchtext的顺序以适应我们"""
    src = batch.src.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# 4.3 多GPU训练
# MultiGPULossCompute,放在了tranformer中

def multi_gpu_test():
    """创建模型，损失函数，优化器，数据迭代器和并行化。"""
    # 由于当前设备并无可用的GPU，因此将调用GPU的代码都注释掉了
    devices = [0, 1, 2, 3]
    device = None  # GPU时该值为0
    if True:
        pad_idx = TGT.vocab.stoi["<blank>"]  # 填充内容编码
        # 构建模型
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        # model.cuda()  # 调用GPU
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx,
                                   smoothing=0.1)
        # criterion.cuda()  # 调用GPU
        BATCH_SIZE = 12000
        # 构建训练数据与测试数据，使用train=True/False区分
        # 设置参数包含，数据，批次宽度，设备，是否重复，
        # 排序方式（先根根据src长度排序，再根据结果trg排序）,
        # 计算所有批次的最大宽度的方法，
        # 是否是训练数据
        train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                                device=device, repeat=False,
                                sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn,
                                train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                                device=device, repeat=False,
                                sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn,
                                train=False)
        # 并行计算
        # model_par = nn.DataParallel(model, device_ids=devices)


# 4.4 训练系统
def multi_gpu_train_system(pad_idx, train_iter, valid_iter, model_par, model, criterion, devices):
    """多GPU并发训练模型的方式，这里并未使用"""
    is_train = False
    if is_train:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      MultiGPULossCompute(model.generator, criterion,
                                          devices, opt=model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             MultiGPULossCompute(model.generator, criterion,
                                                 devices=devices, opt=None))
            print(loss)
        else:
            model = torch.load("../../data/英语-德语翻译/iwslt.pt")


def train_system(pad_idx, train_iter, valid_iter, model, criterion):
    """CPU方式训练系统"""
    is_train = False
    if is_train:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model,
                      SimpleLossCompute(model.generator, criterion, opt=model_opt))
            model.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model,
                             SimpleLossCompute(model.generator, criterion, opt=None))
            print(loss)
        else:
            model = torch.load("../../data/英语-德语翻译/iwslt.pt")


def test_translation(valid_iter, model):
    """贪婪解码，并转换成单词输出"""
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        break

# 4.5 附加组件：BPE，搜索，平均
# 4.6 结果
# 4.7 注意力可视化

