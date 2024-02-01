import config
import utilities as utils
from learning_to_simulate.render_rollout import LOG_PATH


def get_dict_n_layer(d, ks):
    vk = d
    for k in ks:
        vk = utils.get_dict(vk, k, "Unk")
        if vk == "Unk":
            return vk
    return vk

def format_params(param):
    parts = param.split("_")
    v1, v2 = parts[0], parts[1]

    if len(v2) == 1:
        v2 = "0%s" % v2
    return "%s_%s" % (v1, v2)
def rescale(vs, c = 100):
    vv = []
    for v in vs:
        v = c * float(v)
        vv.append("%s" % v)
    return vv
def convert_err_log(inp=LOG_PATH, out="%s/err.csv" % config.DATA_DIR):
    dataset_2_re = {}
    l_types = ["test", "train"]
    l_params = set()
    datasets = set()

    f = open(inp)
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split(",")
        if len(parts) == 1:
            break
        # print(parts)
        data_name, conf, values = parts[0], parts[1], ",".join(rescale(parts[2:]))
        sep_idx = conf.index("_")
        ttype = conf[:sep_idx]
        params = format_params(conf[sep_idx + 1:])
        print(data_name, ttype, params, values)
        dataset_re = utils.get_update_dict(dataset_2_re, data_name, {})
        type_re = utils.get_update_dict(dataset_re, ttype, {})
        type_re[params] = values
        l_params.add(params)
        datasets.add(data_name)

    f.close()
    fout = open(out, "w")
    l_datasets = sorted(list(datasets))
    l_params = sorted(list(l_params))
    n_size = len(l_datasets) * 3
    fout.write(",Params,Dataset%s\n" % (",".join(["" for _ in range(n_size - 1)])))
    fout.write(",,%s\n" % (",,,".join(l_datasets)))
    fout.write(",,%s\n" % ",".join(["MSE,OptimalTrans,CompAvgDist@4" for _ in range(len(l_datasets))]))
    print(l_params)
    for ttype in l_types:
        for i, param in enumerate(l_params):
            if i == 0:
                fout.write("%s,%s" % (ttype, param))
            else:
                fout.write(",%s" % param)
            for data_name in l_datasets:
                fout.write(",%s" % get_dict_n_layer(dataset_2_re, [data_name, ttype, param]))
            fout.write("\n")

    fout.close()


if __name__ == "__main__":
    convert_err_log()
