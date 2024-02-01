import numpy as np
from skbio.stats.distance import mantel
import ot
from sklearn.metrics import mean_squared_error as mse
from scipy import spatial
from scipy.spatial import distance
import config
from scipy.stats import pearsonr, spearmanr
from sklearn.utils.validation import check_symmetric

N_NEIGHBOR_DIF = 4


def __get_mean_closest(pos0, n_poit=N_NEIGHBOR_DIF):
    dis = spatial.distance_matrix(pos0, pos0)
    dis[dis == 0] = 1

    dis = np.sort(dis, axis=1)
    vv = dis[:, :n_poit]

    return np.mean(vv)


def __get_dis_mat_topk(pos0, adj=None, k=4):
    dis = spatial.distance_matrix(pos0, pos0)

    if adj is None:
        if k == -1:
            adj = np.ones((dis.shape[0], dis.shape[0]))
        else:
            # dis[dis == 0] = 10
            adjx = np.argsort(dis, axis=1)[:, :k]
            rids = np.arange(0, adjx.shape[0])
            ridsx = np.tile(rids, (k, 1))
            ridsx = ridsx.transpose().reshape(-1)
            adjx = adjx.reshape(-1)

            adj = np.zeros((dis.shape[0], dis.shape[0]))
            adj[(ridsx, adjx)] = 1
    return dis * adj, adj, dis


def get_diff_neighbor_topk(pos0, pos1, k=4):
    dis1, adj, _ = __get_dis_mat_topk(pos0, k=k)
    dis2, _, _ = __get_dis_mat_topk(pos1, adj, k=k)
    err = dis1 - dis2
    err = err * err
    if config.SQRT:
        err = np.sum(np.sqrt(np.sum(err, axis=1))) / (k * pos0.shape[0])
    else:
        err = np.sum(err) / (k * pos0.shape[0])

    return err


def __check_sym(m):
    return (m == m.T).all()


def get_diff_neighbor_topk2(pos0, pos1, k=-1, tri=True):
    _, adj1, dis1 = __get_dis_mat_topk(pos0, k=k)
    _, adj2, dis2 = __get_dis_mat_topk(pos1, k=k)

    adj = adj1 + adj2
    adj[adj > 0] = 1

    # mts, _, _ = mantel(dis1, dis2)
    mts = -1
    dis1 = (dis1 * adj)
    dis2 = (dis2 * adj)

    if k == -1 and tri:
        triu1 = np.triu_indices(adj.shape[0], 1)
        dis1 = dis1[triu1]
        dis2 = dis2[triu1]
    else:
        dis1 = dis1.reshape(-1)
        dis2 = dis2.reshape(-1)

    ps, _ = pearsonr(dis1, dis2)
    sp, _ = spearmanr(dis1, dis2)

    return [ps, sp, mts]


def get_diff_mean_closet(mata, matb):
    return np.abs(__get_mean_closest(mata) - __get_mean_closest(matb))


def get_ot_m(mata, matb):
    metric = 'sqeuclidean'
    if config.SQRT:
        metric = 'euclidean'
    M = ot.dist(mata, matb, )
    na = mata.shape[0]
    nb = matb.shape[0]
    a, b = ot.unif(na), ot.unif(nb)
    return ot.sinkhorn2(a, b, M / M.max(), reg=0.01)


def getmmd(x, y, sigma=0.1):
    n, d = x.shape
    m, d2 = y.shape
    assert d2 == d
    xy = np.r_[x, y]
    dists = distance.cdist(xy, xy)
    dists /= dists.max()
    k = np.exp((-1 / (2 * sigma ** 2)) * dists ** 2) + np.eye(n + m) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd


def get_p_epsilon(mata, matb, e):
    d = mata - matb
    d = np.sqrt(d * d)
    d = d <= e
    return np.sum(d) / (d.shape[0] * d.shape[1])


def get_all_dist(mata, matb, to_string=True, both=False):
    d1, d2, d3 = mse(mata, matb, squared=not config.SQRT), -1, get_diff_neighbor_topk(mata, matb)
    s = "RMSE: %5.3f, OptimalTrans: %5.3f, CompDistAvg@%s: %5.3f" % (d1, d2, N_NEIGHBOR_DIF, d3)
    if both:
        return s, d1, d2, d3
    elif to_string:
        return s
    return d1, d2, d3


def get_all_dist2(mata, matb, to_string=True, both=False, e=4):
    d1, d2, d3 = mse(mata, matb, squared=not config.SQRT), get_ot_m(mata, matb), get_diff_neighbor_topk(mata, matb)
    d4, d5 = get_p_epsilon(mata, matb, 1e-1), get_p_epsilon(mata, matb, e)
    s = "RMSE: %5.3f, OptimalTrans: %5.3f, CompDistAvg@%s: %5.3f, P@0: %.3f, P@%s %.3f" % (
        d1, d2, N_NEIGHBOR_DIF, d3, d4, e, d5)

    if both:
        return s, d1, d2, d3, d4, d5
    elif to_string:
        return s
    return d1, d2, d3, d4, d5


def get_all_dist3(mata, matb, to_string=True, both=False, n=4, cor=True):
    d1, d2, d3 = mse(mata, matb, squared=not config.SQRT), get_ot_m(mata, matb), get_diff_neighbor_topk(mata, matb)
    dsx = [get_p_epsilon(mata, matb, 2 * i + 1e-1) for i in range(n)]
    pear, _, _ = get_diff_neighbor_topk2(mata, matb, k=-1)
    # s = "RMSE: %5.3f, NeiDis@%s: %5.3f, P-RSE@2: %.3f, P-RSE@6: %.3f, P-RSE@10: %.3f" % (
    #    d1, N_NEIGHBOR_DIF, d3, dsx[1], dsx[3], dsx[5])
    if cor:
        s = "Cor: %.5f" % pear
    else:
        s = "P@10: %.5f" % (dsx[5])
    if both:
        return s, d1, d2, d3, dsx
    elif to_string:
        return s
    return d1, d2, d3, dsx


def get_all_dist4(mata, matb, to_string=True, both=False, n=10, ks=[-1, 4], cor=True):
    d1, d2, d3 = mse(mata, matb, squared=not config.SQRT), -1, -1  # get_ot_m(mata, matb), getmmd(mata, matb)

    dsx = [get_p_epsilon(mata, matb, 2 * i + 1e-1) for i in range(n)]

    dsn = [get_diff_neighbor_topk2(mata, matb, k=ki) for ki in ks]

    # s = "RMSE: %5.3f, P-RSE@10: %.3f, Pearson: %.3f, OT: %.3f" % (d1,  dsx[5], dsn[0][0], d2)
    if cor:
        s = "%.5f" % dsn[1][0]
    else:
        s = "RMSE: %5.3f, P-RSE@10: %.3f, Pearson: %.3f, OT: %.3f" % (d1, dsx[5], dsn[0][0], d2)
    if both:
        return s, d1, d2, d3, dsx, dsn
    elif to_string:
        return s
    return d1, d2, d3, dsx


if __name__ == "__main__":
    np.random.seed(1)
    mata = np.random.random((10, 2)) + 100
    mata2 = np.random.random((10, 2))
    mata3 = mata + 10 + np.random.normal(0, 0.1, (10, 2))

    k = 4

    s, d1, d2, d3, dsx1, dsn = get_all_dist4(mata, mata3, ks=[-1, 4], both=True)
    print(s)
    print(d1)
    print(d2)
    print(d3)
    print(dsx1)
    print(dsn)

    # indices = [(0, 0), (1, 1), (2, 0), (3, 1)]
    # ss = tuple(np.transpose(indices))
    # print(ss)
