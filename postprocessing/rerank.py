# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import numpy as np
import mytools

def acc_i2t2_train(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2_train(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = input[index]
        # Score
        rank = 1e20
        for i in range(5 * index, min(5 * index + 5, image_size*5), 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = input[5 * index + i]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def i2t_rerank(sim, K1):

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def i2t_rerank_optim(sim, K1, Wp1=1.0, Wp2=0.7):

    # print(K1, Wp1, Wp2)

    K1 = max(10, K1)

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    new_sims = np.zeros_like(sim, dtype=np.float32)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    # sort_i2t_re = np.copy(sim)

    for i in range(size_i):
        for j in range(K1):

            # p1 显著性分量
            all_prob = np.sum(sim[:, sort_i2t[i][j]])
            p1 = sim[i][sort_i2t[i][j]] / all_prob
            new_sims[i][sort_i2t[i][j]] = Wp1 * p1

            # p2 原始sim矩阵中排名位置
            p2 = np.exp(-0.05 * (j + 1)) # 归一化
            new_sims[i][sort_i2t[i][j]] += p2


            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]   # 取出每个候选文本对应的所有最优图像

            tmp = np.where(query == i)[0][0]
            address = np.append(address, tmp)    #得到 图像i 使用文本j索引时 所在的位置

        sort = np.argsort(address)
        address = np.array([])

        rank = sort_i2t_re[i][sort]

        # p3 使用候选句查询时的图像排名位置
        for idx, tmp in enumerate(rank):
            p3 = np.exp(-0.05 * (float(idx) + 1))  # 归一化
            new_sims[i][tmp] +=  Wp2 * p3

    return new_sims

def t2i_rerank_optim(sim, K1, Wp1=1.0, Wp2=0.7):
    sim = np.transpose(sim)
    sim = i2t_rerank_optim(sim, K1, Wp1=Wp1, Wp2=Wp2)
    sim = np.transpose(sim)
    return sim


def t2i_rerank(sim, K1):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]

            # ranks = 1e20
            # for k in range(5):
            #     tmp = np.where(query == i//5 * 5 + k)[0][0]
            #     if tmp < ranks:
            #         ranks = tmp
            # address = np.append(address, ranks)
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i

def calc_acc(last_sims):
    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(last_sims)

    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(last_sims)

    currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
    return all_score,currscore

def compare(last_sims, K, Wp1, Wp2):
    _, score = calc_acc(last_sims)
    print("src score:\n {}".format(_))

    sort_rerank = i2t_rerank(last_sims, K1=K)
    (r1i2, r5i2, r10i2, medri2, meanri2), _ = acc_i2t2(sort_rerank)
    # print(r1i2, r5i2, r10i2, np.mean([r1i2, r5i2, r10i2]))
    sort_rerank = t2i_rerank(last_sims, K1=K)
    (r1t2, r5t2, r10t2, medrt2, meanrt2), _ = acc_t2i2(sort_rerank)
    rerank_score = (r1t2 + r5t2 + r10t2 + r1i2 + r5i2 + r10i2) / 6.0
    rerank_scores = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i2, r5i2, r10i2, medri2, meanri2, r1t2, r5t2, r10t2, medrt2, meanrt2, rerank_score
    )
    print("\nrerank score:\n {}".format(rerank_scores))

    sort_rerank = i2t_rerank_optim(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(sort_rerank)
    # print(r1i, r5i, r10i, np.mean([r1i, r5i, r10i]))
    sort_rerank = t2i_rerank_optim(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(sort_rerank)
    optim_score = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0
    optim_scores = "\nOptim: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, optim_score
    )
    print("\noptim score:\n {}".format(optim_scores))


    return score, rerank_score, optim_score

if __name__ == "__main__":
    # ave
    K = 30
    Wp1, Wp2 = 0.30, 1.1

    last_sims = mytools.load_from_npy("file/rsicd.npy")
    score, rerank_score, optim_score = compare(last_sims, K, Wp1, Wp2)


