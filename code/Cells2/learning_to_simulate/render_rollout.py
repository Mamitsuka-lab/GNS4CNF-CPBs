# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m learning_to_simulate.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""  # pylint: disable=line-too-long
import copy
import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from PIL import Image
import config
from learning_to_simulate import dist
import cv2

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_string("out_path", None, help="Path to output file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")
flags.DEFINE_integer("test_id", 0, help="Id of test file (0,1,2,3)")
flags.DEFINE_integer("rollout_offset", 0, help="Rollout offset")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
    1: "red",  # Water.
}
DARK_MODE = True
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import animation, colormaps
from PIL import Image
from matplotlib.patches import Rectangle

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph

LOG_PATH = "%s/log_erros_rollout1.txt" % config.DATA_DIR
LOG_PATH_A = "%s/log_erros_rollout1_all_steps.txt" % config.DATA_DIR


def make_eps_neighborhood_graph(pos, eps):
    A = (squareform(pdist(pos)) < eps).astype(int)
    # A[A==0] = 1000
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_matrix(A)
    return G


def make_knn_graph(pos, k=5):
    A = kneighbors_graph(pos, k).toarray()
    # A[A==0] = 1000
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_matrix(A)
    return G


def main3(unused_argv):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    if not FLAGS.out_path:
        raise ValueError("A `out_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)
    # plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = 'black'

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="black")

    plot_info = []
    trajectories = []
    labels = ["Ground truth @ ", "Prediction @ "]
    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]], axis=0)
        print("Traject shape: ", trajectory.shape)
        trajectories.append(trajectory)
        ax = axes[ax_i]
        # ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        # points = {
        #     particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
        #    for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, trajectory))

    num_steps_ground_truth = trajectories[0].shape[0]
    num_steps_rollouts = trajectories[1].shape[0]
    num_steps_l = [num_steps_ground_truth, num_steps_rollouts]
    min_x = min(num_steps_rollouts, num_steps_ground_truth)
    print("Num steps: ", num_steps_ground_truth, num_steps_rollouts)
    bounds = rollout_data["metadata"]["bounds"]

    def get_step(i, j):
        v = num_steps_l[j]
        if i > v - 1:
            i = v - 1
        return i

    def update(step_i):
        print("\r@Step: ", step_i, end="")
        for j, (ax, trajectory) in enumerate(plot_info):
            # print("J", j)
            step_ii = get_step(step_i, j)
            ax.clear()
            if DARK_MODE:
                ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, color='black', alpha=0.7))
            # ax.set_facecolor('black')
            ax.set_xlim(bounds[0][0], bounds[0][1])
            ax.set_ylim(bounds[1][0], bounds[1][1])
            ax.set_title("%s%s" % (labels[j], step_ii + 1))
            # print("LPOINT ", len(points.items()))
            G = make_eps_neighborhood_graph(trajectory[step_ii],
                                            rollout_data["metadata"]["default_connectivity_radius"])
            # G = make_knn_graph(trajectory[step_ii], k=4)
            dpos = {"x": trajectory[step_ii, :, 0], "y": trajectory[step_ii, :, 1]}
            n_node = trajectory[step_ii].shape[0]
            if DARK_MODE:
                nx.draw(
                    G, trajectory[step_ii], ax=ax, node_size=50, node_color=range(n_node), cmap=plt.get_cmap('hsv'),
                    edge_color="white", alpha=0.9
                )
            else:
                nx.draw(G, trajectory[step_ii], ax=ax, node_size=50, node_color="red", edge_color="white", alpha=0.3)

            # ax.set_axis_off()

        if step_i <= min_x - 1:
            dis_text, d1, d2, d3 = dist.get_all_dist(trajectories[0][step_i], trajectories[1][step_i], both=True)
            fig.suptitle('Error @%3s: %s' % (step_ii + 1, dis_text))
            if step_i == min_x - 1:
                parts = FLAGS.out_path.split("/")
                data_name, conf = parts[-2], parts[-1]
                fout = open(LOG_PATH, "a")
                fout.write("%s,%s,%.5f,%.5f,%.5f\n" % (data_name, conf, d1, d2, d3))
                fout.close()

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(config.C - 1, max(num_steps_l) + 1, FLAGS.step_stride), interval=1000)
    # frames=np.arange(config.C - 1,config.C + 5, FLAGS.step_stride), interval=1000)

    # plt.show(block=FLAGS.block_on_show)
    writergif = animation.PillowWriter(fps=2)
    unused_animation.save("%s_graph_dark.gif" % FLAGS.out_path, writergif, savefig_kwargs={'facecolor': 'auto'})

    # FFwriter = animation.FFMpegWriter(fps=2)
    # unused_animation.save("%s/Demo2.mp4" % config.TEST1_DIR,FFwriter)


def main2(unused_argv):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    if not FLAGS.out_path:
        raise ValueError("A `out_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_info = []
    trajectories = []
    labels = ["Ground truth @ ", "Prediction @ "]
    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]], axis=0)
        print("Traject shape: ", trajectory.shape)
        trajectories.append(trajectory)
        ax = axes[ax_i]
        # ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        points = {
            particle_type: ax.plot([], [], "o", markersize=5, color=color, alpha=0.3)[0]
            for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, trajectory, points))

    num_steps_ground_truth = trajectories[0].shape[0]
    num_steps_rollouts = trajectories[1].shape[0]
    num_steps_l = [num_steps_ground_truth, num_steps_rollouts]
    min_x = min(num_steps_rollouts, num_steps_ground_truth)
    print("Num steps: ", num_steps_ground_truth, num_steps_rollouts)

    def get_step(i, j):
        v = num_steps_l[j]
        if i > v - 1:
            i = v - 1
        return i

    def update(step_i):
        outputs = []
        for j, (_, trajectory, points) in enumerate(plot_info):
            # print("J", j)
            step_ii = get_step(step_i, j)
            axes[j].set_title("%s%s" % (labels[j], step_ii + 1))
            # print("LPOINT ", len(points.items()))
            for particle_type, line in points.items():
                mask = rollout_data["particle_types"] == particle_type
                # print(trajectory[step_i, mask, 0])
                line.set_data(trajectory[step_ii, mask, 0],
                              trajectory[step_ii, mask, 1])

                outputs.append(line)
        if step_i <= min_x - 1:
            dis_text, d1, d2, d3 = dist.get_all_dist(trajectories[0][step_i], trajectories[1][step_i], both=True)
            fig.suptitle('Error @%3s: %s' % (step_ii + 1, dis_text))
            if step_i == min_x - 1:
                parts = FLAGS.out_path.split("/")
                data_name, conf = parts[-2], parts[-1]
                fout = open(LOG_PATH, "a")
                fout.write("%s,%s,%.5f,%.5f,%.5f\n" % (data_name, conf, d1, d2, d3))
                fout.close()
        return outputs

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(config.C - 1, max(num_steps_l) + 1, FLAGS.step_stride), interval=1000)
    # plt.show(block=FLAGS.block_on_show)
    writergif = animation.PillowWriter(fps=3)
    unused_animation.save("%s.gif" % FLAGS.out_path, writergif)
    # FFwriter = animation.FFMpegWriter(fps=2)
    # unused_animation.save("%s/Demo2.mp4" % config.TEST1_DIR,FFwriter)


def save_gifanim(np_im_list, outfile, dur=800):
    im_list = []
    for x in np_im_list:
        im = Image.fromarray(x).convert("P")
        im_list.append(im)
    im_list[0].save(
        outfile, save_all=True, append_images=im_list[1:], optimize=False, duration=dur
    )


def restore(pos, bmin, bmax, minx, miny, maxx, maxy, dd, cp=False):
    if cp:
        pos = copy.deepcopy(pos)
    pos[:, 0] -= bmin
    pos[:, 0] /= (bmax - bmin)
    pos[:, 0] *= (maxx - minx)
    pos[:, 0] += minx

    pos[:, 1] -= bmin
    pos[:, 1] /= (bmax - bmin)
    pos[:, 1] *= (maxy - miny)
    pos[:, 1] += miny

    s1 = (maxx - minx) / (bmax - bmin)
    s2 = (maxy - miny) / (bmax - bmin)
    dd2 = np.sqrt((s1 * s1 + s2 * s2) / 2) * dd
    return dd2, pos


labeloff = {'labelbottom': False,
            'labelleft': False,
            'labelright': False,
            'labeltop': False}


def add_margin(pil_img, top, right, bottom, left, color="black"):
    print("PIL SHAPE", pil_img.shape)
    width, height = pil_img.shape
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def load_imgs(path):
    videos = cv2.VideoCapture(path)
    imgs = []
    while True:
        ret, frame = videos.read()
        if ret:
            imgs.append(frame)
        else:
            break
    return imgs


def get_db1(ll):
    s = np.vstack(ll)
    v = s[1:] - s[:-1]
    a = v
    # a = v[1:] - v[:-1]
    # a = a[1:] - a[:-1]
    a0 = a[:, :, 0]
    a1 = a[:, :, 1]
    aa = np.sqrt(a0 * a0 + a1 * a1)
    aa = np.mean(aa, axis=1)
    return [np.mean(aa), np.std(aa)]


def get_db2(ll):
    s = np.vstack(ll)
    v = s[1:] - s[:-1]
    # a = v
    a = v[1:] - v[:-1]
    # a = a[1:] - a[:-1]
    a0 = a[:, :, 0]
    a1 = a[:, :, 1]
    aa = np.sqrt(a0 * a0 + a1 * a1)
    aa = np.mean(aa, axis=1)
    return [np.mean(aa), np.std(aa)]


def main4(unused_argv):
    print("Rollout path", FLAGS.rollout_path)

    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    if not FLAGS.out_path:
        raise ValueError("A `out_path` must be passed.")
    with open("%s" % FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)
    # plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    data_name = FLAGS.out_path.split("/")[-2]
    video_path = "%s/%s/%s.avi" % (config.DATA_DIR, data_name, FLAGS.test_id)
    vid = imageio.get_reader(video_path, "ffmpeg")
    img_seq = [image for image in vid]
    # img_seq = load_imgs(video_path)
    assert rollout_data["metadata"]['r_steps'] == len(img_seq)
    plot_info = []
    trajectories = []
    labels = ["Ground truth @ ", "Prediction @ "]
    l_test = []
    l_train = []

    parts = FLAGS.out_path.split("/")
    data_name, conf = parts[-2], parts[-1]

    fout_a = open(LOG_PATH_A, "a")
    kcs = [-1, 4, 8]
    metadata = rollout_data["metadata"]
    bounds = metadata["bounds"]

    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]], axis=0)
        print("Traject shape: ", trajectory.shape)
        trajectories.append(trajectory)
        ax = axes[ax_i]
        # ax.set_title(label)
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        # points = {
        #     particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
        #    for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, trajectory))

    num_steps_ground_truth = trajectories[0].shape[0]
    num_steps_rollouts = trajectories[1].shape[0]
    num_steps_l = [num_steps_ground_truth, num_steps_rollouts]
    min_x = min(num_steps_rollouts, num_steps_ground_truth)
    print("Num steps: ", num_steps_ground_truth, num_steps_rollouts)
    metadata = rollout_data["metadata"]
    print("Bounds", metadata['bounds'])

    img_frames = []

    def check_offset(v):
        if config.OFF_SET > 0 and not config.TEST_ALL:
            v = v - config.C + config.OFF_SET + FLAGS.rollout_offset
        return v

    res = None
    start_pos = None
    ran_dom_p = None
    print("R lenght: ", len(trajectories[1]), FLAGS.rollout_offset, len(img_seq))
    cor_lists = []
    OFFSET = 0
    for i, im in enumerate(img_seq[OFFSET + FLAGS.rollout_offset:]):
        pref = "Initialization"
        color = "green"

        posp = trajectories[1][i]
        pos0 = trajectories[0][i]
        iFrame = i + OFFSET + FLAGS.rollout_offset

        dd, posp = restore(posp, metadata['bounds'][0][0], metadata["bounds"][0][1],
                     metadata["min_x"], metadata["min_y"],
                     metadata["max_x"], metadata["max_y"],
                     metadata["default_connectivity_radius"], cp=True
                     )


        G = make_eps_neighborhood_graph(posp, dd)

        _, pos0 = restore(pos0, metadata['bounds'][0][0], metadata["bounds"][0][1],
                metadata["min_x"], metadata["min_y"],
                metadata["max_x"], metadata["max_y"],
                metadata["default_connectivity_radius"], cp=True
                )
        dx, dy = metadata["max_x"] - metadata["min_x"], metadata["max_y"] - metadata["min_y"]
        if start_pos is None:
            start_pos = pos0
            nr, nc = pos0.shape
            v = pos0.reshape(-1)
            std = np.std(v)
            noise = np.random.normal(v, std / 10)
            ran_dom_p = start_pos + noise.reshape((nr, nc))
        dis_text, d1, d2, d3, dsx, dsx2 = dist.get_all_dist4(pos0, posp, both=True, n=config.N_PN, ks=kcs)
        cor_lists.append(dis_text)
        fout_a.write("%s,%s, %s, %s, %.5f,%.5f,%.5f" % (data_name, FLAGS.test_id, FLAGS.rollout_offset, check_offset(iFrame + 1), d1, d2, d3))
        for jj in range(config.N_PN):
            fout_a.write(",%.5f" % dsx[jj])
        for jj in range(len(kcs)):
            for jk in range(3):
                fout_a.write(",%.5f" % dsx2[jj][jk])
        fout_a.write("\n")

        fig, ax = plt.subplots()
        if iFrame >= config.OFF_SET + FLAGS.rollout_offset:
            pref = "Prediction"
            color = "blue"
        if iFrame >= config.OFF_SET + FLAGS.rollout_offset - 1:
            l_test.append(pos0[None])
        else:
            l_train.append(pos0[None])

        nx.draw(
            G, posp, ax=ax, node_size=50, node_color=color, edge_color="white", alpha=0.6
        )

        im = im[20:-20, 20:-20]
        pim = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=0)

        ax.imshow(pim)
        ax.tick_params(**labeloff)
        fig.suptitle('%s  \nCor @Step %3s: %s' % (pref,  check_offset(iFrame + 1), dis_text),
                     color=color, fontsize=8)

        ax.set_axis_off()
        fig.tight_layout()
        fig.canvas.draw()

        x = np.asarray(fig.canvas.buffer_rgba())
        img_frames.append(x)
        plt.close()

        if iFrame == len(img_seq) - 1:

            # dba = "$".join(["%s" % aai for aai in aa])
            # dba_train1 = "$".join(["%.5f" % vi for vi in get_db1(l_train)])
            # dba_train2 = "$".join(["%.5f" % vi for vi in get_db2(l_train)])
            dba_train1 = ""
            dba_train2 = ""
            dba_test1 = "$".join(["%.5f" % vi for vi in get_db1(l_test)])
            dba_test2 = "$".join(["%.5f" % vi for vi in get_db2(l_test)])

            _, d1s, d2s, d3s, dsxs, dsxs2 = dist.get_all_dist4(pos0, start_pos, both=True, n=config.N_PN, ks=kcs)

            _, d1n, d2n, d3n, dsxn, dsxn2 = dist.get_all_dist4(pos0, ran_dom_p, both=True, n=config.N_PN, ks=kcs)

            fout = open(LOG_PATH, "a")
            fout.write("%s,%s,%s, %s, %.5f,%.5f,%.5f" % (data_name, "%s#GNet" % conf, FLAGS.test_id, FLAGS.rollout_offset, d1, d2, d3))
            for jj in range(config.N_PN):
                fout.write(",%.5f" % dsx[jj])
            for jj in range(len(kcs)):
                for jk in range(3):
                    fout.write(",%.5f" % dsx2[jj][jk])
            fout.write(",%s,%s,%s,%s\n" % (dba_train1, dba_test1, dba_train2, dba_test2))


            fout.write("%s,%s,%s,%s, %.5f,%.5f,%.5f" % (
                data_name, "%s#Static" % conf, FLAGS.test_id, FLAGS.rollout_offset, d1s, d2s, d3s))

            for jj in range(config.N_PN):
                fout.write(",%.5f" % dsxs[jj])
            for jj in range(len(kcs)):
                for jk in range(3):
                    fout.write(",%.5f" % dsxs2[jj][jk])
            fout.write("\n")

            fout.write("%s,%s,%s,%s,%.5f,%.5f,%.5f" % (
                data_name, "%s#Random" % conf, FLAGS.test_id, FLAGS.rollout_offset, d1n,d2n, d3n))

            for jj in range(config.N_PN):
                fout.write(",%.5f" % dsxn[jj])
            for jj in range(len(kcs)):
                for jk in range(3):
                    fout.write(",%.5f" % dsxn2[jj][jk])
            fout.write("\n")

            fout.close()

    save_gifanim(img_frames, "%s_%s_%s.gif" % (FLAGS.out_path, FLAGS.test_id, FLAGS.rollout_offset))
    fout_a.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_info = []
    labels = ["Ground truth @ ", "Prediction @ "]
    print("Cor_list", len(cor_lists))
    def __revert_id(step_i):
        return step_i - OFFSET  - (FLAGS.rollout_offset + 1)
    def update(step_i):
        ii = __revert_id(step_i)
        color = "green"


        # print(step_i,ii)
        posp = trajectories[1][ii]
        pos0 = trajectories[0][ii]
        Gp = make_eps_neighborhood_graph(posp, metadata["default_connectivity_radius"])
        G0 = make_eps_neighborhood_graph(pos0, metadata["default_connectivity_radius"])
        Gs = [G0, Gp]
        n_node = pos0.shape[0]
        ps = [pos0, posp]
        pref = "Initialization"

        for ia, ax in enumerate(axes):
            ax.clear()
            ax.set_xlim(bounds[0][0], bounds[0][1])
            ax.set_ylim(bounds[1][0], bounds[1][1])
            lb = labels[ia]
            if ia == 0:
                color = "green"
            else:
                if step_i >= config.C + FLAGS.rollout_offset + 1:
                    color = "blue"
                    pref = "Prediction"
                else:
                    lb = labels[0]
            ax.set_title("%s%s" % (lb, step_i), color=color)
            ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, color='black', alpha=0.7))

            nx.draw(
                Gs[ia], ps[ia], ax=ax, node_size=50, node_color=range(n_node), cmap=plt.get_cmap('hsv'),
                edge_color="white", alpha=0.9
            )
            fig.suptitle('Cor @%3s: %s' % (step_i, cor_lists[ii]), color=color)

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(OFFSET + FLAGS.rollout_offset + 1, len(img_seq) + 1, FLAGS.step_stride), interval=1000)
    # frames=np.arange(config.C - 1,config.C + 5, FLAGS.step_stride), interval=1000)

    # plt.show(block=FLAGS.block_on_show)
    writergif = animation.PillowWriter(fps=2)
    unused_animation.save("%s_%s_%s_sides.gif" % (FLAGS.out_path, FLAGS.test_id, FLAGS.rollout_offset), writergif, savefig_kwargs={'facecolor': 'auto'})

if __name__ == "__main__":
    # app.run(main2)
    app.run(main4)
