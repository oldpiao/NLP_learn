from transformer import *


def run_test():
    # 训练一个简单的拷贝任务
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        ))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))


if __name__ == '__main__':
    # run_test()
    V = 10  # src_vocab, tgt_vocab, 测试数据设计的输入输出相同
    model = make_model(V, V, 6)
    batch = next(data_gen(V, 20, 1))
    greedy_decode(model, batch.src, batch.src_mask, max_len=10, start_symbol=0)
