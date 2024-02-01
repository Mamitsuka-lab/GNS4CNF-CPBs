import config
import tensorflow as tf
import glob
import numpy as np
import utilities
import json
import os


def load_csv(path):
    print(path)
    f = open(path, errors='ignore')
    # Skip header
    f.readline()
    slice_2_points = {}
    max_x = max_y = -1
    min_x = min_y = 1e10

    # Re-assign point ids using a dict
    point_no_2_id = dict()
    previous_point_no = 1
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split(",")
        if len(parts[1]) == 0:
            break

        if len(parts[0]) == 0:
            point_no = previous_point_no
        else:
            point_no = int(parts[0])
            previous_point_no = point_no
        point_id = utilities.get_insert_indices_dict(point_no_2_id, point_no)
        slice_id = int(parts[1])
        x = int(parts[2])
        y = int(parts[3])
        slice_points = utilities.get_update_dict(slice_2_points, slice_id, {})
        slice_points[point_id] = [x, y]
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    n_slices = len(slice_2_points)
    n_points = len(slice_2_points[1])
    slice_2_point_array = {}

    def get_slice_point(slice_points2, point_id2, prev_slice_points):
        # Get point position from previous slice if not exist in current slice
        xx, yy = utilities.get_dict(slice_points2, point_id2, (-1, -1))
        if xx == -1:
            if prev_slice_points is not None:
                xx, yy = utilities.get_dict(prev_slice_points, point_id2, (-1, -1))
        return xx, yy

    for i in range(n_slices):
        slice_points = slice_2_points[i + 1]
        if i == 0:
            prev_slices = None
        else:
            prev_slices = slice_2_points[i]
        ar = np.ndarray((n_points, 2), dtype=np.float32)
        for j in range(n_points):
            x, y = get_slice_point(slice_points, j, prev_slices)
            ar[j, 0] = x
            ar[j, 1] = y
        slice_2_point_array[i] = ar
    print("Load csv", path, n_slices, n_points, max_y, max_y)
    return slice_2_point_array, n_slices, n_points, max_x, max_y, min_x, min_y


def make_sequential_example3(tp, key=0, particle_type_default=1, poss=None, serialized=True):
    npoint = poss[0].shape[0]
    seq_len1 = len(poss)
    assert seq_len1 == 25
    ctxt = np.ndarray([seq_len1, 1], dtype=np.float32)

    ctxt.fill(config.ENVS[tp])

    feature_list = {"position": tf.train.FeatureList(
        feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.reshape(-1).tobytes()])) for feature in
                 poss]),
        "step_context": tf.train.FeatureList(
            feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.reshape(-1).tobytes()])) for feature
                     in
                     ctxt])
    }
    context_feature = {"key": tf.train.Feature(int64_list=tf.train.Int64List(value=[key])),
                       "particle_type":
                           tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                               np.asarray([particle_type_default for _ in range(npoint)], dtype=np.int64).tobytes()]))}

    context = tf.train.Features(feature=context_feature)
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    # print(context)
    # print(feature_lists)
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    if serialized:
        example = example.SerializeToString()

    return example


def convert_2_sequence_example(slice_2_point_array, tp, serilized=False):
    assert type(slice_2_point_array) == np.ndarray
    example = make_sequential_example3(poss=slice_2_point_array, tp=tp, serialized=serilized)
    return example


def write_tfrecord(bytes, file):
    with tf.io.TFRecordWriter(file) as writer:
        writer.write(bytes)


MIN_B = 0.1
MAX_B = 0.9


def load_files(dir_path, data_pref):
    print("Loading...", dir_path)
    files = sorted(glob.glob("%s/*.csv" % dir_path))
    filesv = sorted(glob.glob("%s/*.avi" % dir_path))

    parts = dir_path.split("/")
    print(parts)
    etype = config.TYPE_2_INT[parts[-2].split("_")[-1]]
    etype_offset = etype * config.TYPE_SIZE
    # rename
    for i, filep in enumerate(filesv):
        cmd = "cp \"%s\" %s/%s_%s.avi" % (filep, config.DATA_DIR, data_pref, etype_offset + i)
        print(cmd)
        os.system(cmd)  # print(files)

    file_ind_2_data = dict()
    for i, file in enumerate(files):
        print(file)
        # slice_2_point_array, n_slices, n_points, max_x, max_y, min_x, min_y = load_csv(file)
        data_offset = etype_offset + i
        file_ind_2_data[data_offset] = load_csv(file)

    return file_ind_2_data


def dict_2_tensor(d):
    n = len(d)
    ar = []
    for i in range(n):
        ar.append(d[i])
    return np.asarray(ar)


def load_data_files(data_pref):
    ptn = "%s/%s_*/" % (config.DATA_DIR, data_pref)
    print("P", ptn)
    dir_paths = sorted(glob.glob(ptn))

    findices_2_data = dict()

    print("Loading data files..")
    print(dir_paths)
    for dir_path in dir_paths:
        print(dir_path)
        d = load_files(dir_path, data_pref)
        for k, v in d.items():
            findices_2_data[k] = v
    print(len(findices_2_data))
    assert len(findices_2_data) == 12
    minx = 1e10
    miny = 1e10
    maxx = 0
    maxy = 0
    poss3 = []
    for i in range(12):
        slice_2_point_array, n_slices, n_points, max_x, max_y, min_x, min_y = findices_2_data[i]
        pos = dict_2_tensor(slice_2_point_array)
        poss3.append(pos)
        maxx = max(maxx, max_x)
        maxy = max(maxy, max_y)
        minx = min(minx, min_x)
        miny = min(miny, min_y)

    def norm_pos(pos, maxx, minx, maxy, miny):
        pos[:, :, 0] -= minx
        pos[:, :, 0] /= (maxx - minx)
        pos[:, :, 0] *= (MAX_B - MIN_B)
        pos[:, :, 0] += MIN_B

        pos[:, :, 1] -= miny
        pos[:, :, 1] /= (maxy - miny)
        pos[:, :, 1] *= (MAX_B - MIN_B)
        pos[:, :, 1] += MIN_B

    for pos in poss3:
        norm_pos(pos, maxx, minx, maxy, miny)

    def get_diff(vv):
        v_next = vv[1:, :]
        v_curr = vv[:-1, :]
        v_diff = v_next - v_curr
        return v_diff

    vecs = []
    accs = []
    for i, pos in enumerate(poss3[:-1]):
        print(pos.shape)
        vec = get_diff(pos)
        vec = vec.reshape((-1, 2))
        acc = get_diff(vec)
        acc = acc.reshape((-1, 2))
        vecs.append(vec)
        accs.append(acc)
    vecs = np.vstack(vecs)
    accs = np.vstack(accs)
    vec_mean, vec_std = np.mean(vecs, axis=0), np.std(vecs, axis=0)
    acc_mean, acc_std = np.mean(accs, axis=0), np.std(accs, axis=0)
    # print(vec_mean, vec_std)
    # print(acc_mean, acc_std)
    from scipy import spatial

    mean_list = []
    for i in [config.TYPE_SIZE * ii for ii in range(3)]:
        pos0 = poss3[i][0, :]
        # print(pos0.shape)
        dis = spatial.distance_matrix(pos0, pos0)
        dis[dis == 0] = 1

        dis = np.sort(dis, axis=1)
        vv = dis[:, :config.N_POINT_DIS]

        mn = np.mean(vv)
        mean_list.append(mn)

    radius = np.mean(mean_list)

    return poss3, np.float64(vec_mean), np.float64(vec_std), np.float64(acc_mean), np.float64(acc_std), np.float64(
        radius), maxx, minx, maxy, miny


def convert_files(data_pref):


    poss, vec_mean, vec_std, acc_mean, acc_std, top_k_neighbor_mean, maxx, minx, maxy, miny = load_data_files(data_pref)
    n_files = len(poss)
    bmark = False
    print("NUM FILES: ", n_files)
    metainfo = {}
    metainfo["sequence_length"] = len(poss[0]) - 1
    metainfo["dim"] = 2
    metainfo["dt"] = 0.01
    metainfo.update({"bounds": [[MIN_B, MAX_B], [MIN_B, MAX_B]],
                     "vel_mean": [vec_mean[0], vec_mean[1]], "vel_std": [vec_std[0], vec_std[1]],
                     "acc_mean": [acc_mean[0], acc_mean[1]], "acc_std": [acc_std[0], acc_std[1]],
                     "default_connectivity_radius": top_k_neighbor_mean,
                     "r_steps": 25,
                     "max_x": maxx, "max_y": maxy, "min_x": minx, "min_y": miny,
                     "context_mean": [np.mean(config.ENVS)], "context_std": [np.std(config.ENVS)]})
    for i in range(n_files):
        slice_2_point_array = poss[i]
        tp = i // config.TYPE_SIZE
        # print("TYPE: ", tp)
        example = convert_2_sequence_example(slice_2_point_array, tp=tp, serilized=True)
        if i in config.FILE_TEST_IDS:
            write_tfrecord(example, "%s/test_%s_%s.tfrecord" % (config.DATA_DIR, data_pref, i))
        else:
            write_tfrecord(example, "%s/train_%s_%s.tfrecord" % (config.DATA_DIR, data_pref, i))


    print(metainfo)
    with open("%s/metadata_%s.json" % (config.DATA_DIR, data_pref), "w") as f:
        json.dump(metainfo, f)


if __name__ == "__main__":

    if config.CLEAN:
        import os

        os.system("rm -r %s/*" % config.MODEL_DIRS)
        os.system("rm -r %s/*" % config.ROLLOUT_DIRS)
    from converting_data.gen_files import create_bash_script

    create_bash_script("%s/run_1" % config.C_DIR)


    cmd = "rm %s/*.tfrecord" % config.DATA_DIR
    os.system(cmd)
    cmd = "rm %s/*.avi" % config.DATA_DIR
    os.system(cmd)
    for pref in ["hmsc", "chondro"]:

        config.DATA_PREX = pref
        convert_files(data_pref=pref)

