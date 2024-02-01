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

    return slice_2_point_array, n_slices, n_points, max_x, max_y, min_x, min_y


def make_sequential_example(key=0, particle_type_default=1, poss=None, serialized=True):
    npoint = poss[0].shape[0]
    feature_list = {"position": tf.train.FeatureList(
        feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.reshape(-1).tostring()])) for feature in
                 poss])}
    context_feature = {"key": tf.train.Feature(int64_list=tf.train.Int64List(value=[key])),
                       "particle_type":
                           tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                               np.asarray([particle_type_default for _ in range(npoint)], dtype=np.int64).tostring()]))}

    context = tf.train.Features(feature=context_feature)
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    # print(context)
    # print(feature_lists)
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    if serialized:
        example = example.SerializeToString()

    return example


def convert_2_sequence_example_2(slice_2_point_array, n_test= config.N_TEST2,serilized=False,test_all=False):
    assert type(slice_2_point_array) == np.ndarray
    poss_train = slice_2_point_array[:-n_test]
    poss_test = slice_2_point_array[-(n_test+config.C):]
    # print("L Train: ", len(poss_train))
    # print("L Test: ", len(poss_test))
    example_train = make_sequential_example(poss=poss_train, serialized=serilized)
    if test_all:
        example_test = make_sequential_example(poss=slice_2_point_array,serialized=serilized)
    else:
        example_test = make_sequential_example(poss=poss_test, serialized=serilized)

    return example_train, example_test


def write_tfrecord(bytes, file):
    with tf.io.TFRecordWriter(file) as writer:
        writer.write(bytes)


MIN_B = 0.1
MAX_B = 0.9


def load_files(dir_path):
    files = sorted(glob.glob("%s/*.csv" % dir_path))
    filesv = sorted(glob.glob("%s/*.avi" % dir_path))

    # rename
    for i, filep in enumerate(filesv):
        dir = "/".join(filep.split("/")[:-1])
        cmd = "mv \"%s\" %s/%s.avi" % (filep, dir, i)
        print(cmd)
        os.system(cmd)
    # print(files)

    def dict_2_tensor(d):
        n = len(d)
        ar = []
        for i in range(n):
            ar.append(d[i])
        return np.asarray(ar)

    poss = []
    minx = 1e10
    miny = 1e10
    maxx = 0
    maxy = 0
    npoints_l = []
    for i, file in enumerate(files):
        slice_2_point_array, n_slices, n_points, max_x, max_y, min_x, min_y = load_csv(file)
        npoints_l.append(n_points)
        pos = dict_2_tensor(slice_2_point_array)
        poss.append(pos)
        maxx = max(max_x, maxx)
        maxy = max(max_y, maxy)
        minx = min(min_x, minx)
        miny = min(min_y, miny)

    min_frame_id = np.argmin(npoints_l)

    # normalizing
    for pos in poss:
        pos[:, :, 0] -= minx
        pos[:, :, 0] /= (maxx - minx)
        pos[:, :, 0] *= (MAX_B - MIN_B)
        pos[:, :, 0] += MIN_B

        pos[:, :, 1] -= miny
        pos[:, :, 1] /= (maxy - miny)
        pos[:, :, 1] *= (MAX_B - MIN_B)
        pos[:, :, 1] += MIN_B

    # Test set: Last one
    # Get velocity:
    def get_diff(vv):
        v_next = vv[1:, :]
        v_curr = vv[:-1, :]
        v_diff = v_next - v_curr
        return v_diff

    vecs = []
    accs = []
    for i, pos in enumerate(poss[:-1]):
        # print(pos.shape)
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

    pos0 = poss[0][0, :]
    # print(pos0.shape)
    from scipy import spatial
    dis = spatial.distance_matrix(pos0, pos0)
    dis[dis == 0] = 1

    dis = np.sort(dis, axis=1)
    vv = dis[:, :config.N_POINT_DIS]

    top_k_neighbor_mean = np.mean(vv)
    return poss, np.float64(vec_mean), np.float64(vec_std), np.float64(acc_mean), np.float64(acc_std), np.float64(
        top_k_neighbor_mean), min_frame_id, maxx, minx, maxy, miny


def convert_files(dir_path, test_on_train=False, test_all = False):
    poss, vec_mean, vec_std, acc_mean, acc_std, top_k_neighbor_mean, min_frame_id, maxx, minx, maxy, miny = load_files(dir_path)
    # print("Min frame: ", min_frame_id)
    n_files = len(poss)
    bmark = False
    if test_all:
        test_on_train = False
    for i in range(n_files):
        slice_2_point_array = poss[i]
        example_train_i, example_test_i = convert_2_sequence_example_2(slice_2_point_array, serilized=True, test_all=test_all)


        write_tfrecord(example_train_i, "%s/train_%s.tfrecord" % (dir_path, i))
        if test_on_train:
            if not bmark:
                write_tfrecord(example_train_i, "%s/test.tfrecord" % dir_path)
                bmark = True
        else:
            write_tfrecord(example_test_i, "%s/test_%s.tfrecord" % (dir_path,i))


    metainfo = {}
    metainfo["dim"] = 2
    metainfo["dt"] = 0.01
    r_step = config.N_TEST2 + config.C
    if test_all:
        r_step = 25

    # Train Meta Info
    metainfo.update({"bounds": [[MIN_B, MAX_B], [MIN_B, MAX_B]],
                     "vel_mean": [vec_mean[0], vec_mean[1]], "vel_std": [vec_std[0], vec_std[1]],
                     "acc_mean": [acc_mean[0], acc_mean[1]], "acc_std": [acc_std[0], acc_std[1]],
                     "default_connectivity_radius": top_k_neighbor_mean,
                     "r_steps" : r_step,
                     "max_x": maxx, "max_y": maxy, "min_x": minx, "min_y":miny})

    metainfo["sequence_length"] = len(poss[0]) - 1 - config.N_TEST2

    with open("%s/metadata_train.json" % dir_path, "w") as f:
        json.dump(metainfo, f)

    # Test Meta Info
    if test_all:
        metainfo["sequence_length"] = len(poss[0]) - 1
    else:
        metainfo["sequence_length"] = config.N_TEST2 + config.C - 1
    with open("%s/metadata_test.json" % dir_path, "w") as f:
        json.dump(metainfo, f)



if __name__ == "__main__":
    from utilities import ensuredir
    ensuredir(config.MODEL_DIRS)
    ensuredir(config.ROLLOUT_DIRS)
    if config.CLEAN:
        import os
        os.system("rm -r %s/*" % config.MODEL_DIRS)
        os.system("rm -r %s/*" % config.ROLLOUT_DIRS)
    from converting_data.gen_files import create_bash_script
    create_bash_script("%s/run_1" % config.C_DIR)
    for dir_path in glob.glob("%s/*_*/"%config.DATA_DIR):
        convert_files(dir_path=dir_path, test_on_train=False, test_all=True)
    # for dir_path in ["/home/gpux1/Codes/Cells/data/hmsc_high"]:
    #    convert_files(dir_path, test_on_train=True)
