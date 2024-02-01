import glob
import config
import utilities as utils
import os


def create_bash_script(bash_dir):
    # Train script
    path1 = "%s_train.sh" % bash_dir
    params = "%s_%s" % (config.C, config.N_POINT_DIS)
    f_script = open(path1, "w")
    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -ne 2 ];then\n")
    f_script.write("    echo \"Need 2 argument: data_name test_id\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")
    f_script.write("python -m learning_to_simulate.train --test_id=$2 --data_path=%s/$1 --model_path=%s/\"$1\"_%s_\"$2\"\n"
                   % (config.DATA_DIR, config.MODEL_DIRS, params))
    f_script.close()
    os.system("chmod +x %s" % path1)

    # Rollout script
    path2 = "%s_rollout.sh" % bash_dir
    f_script = open(path2, "w")
    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -ne 3 ];then\n")
    f_script.write("    echo \"Need 2 argument: data_name test_id rollout_offset\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")

    f_script.write("python -m learning_to_simulate.train --rollout_offset=$3 --test_id=$2 --mode=\"eval_rollout\" --data_path=%s/$1 "
                   "--model_path=%s/\"$1\"_%s_\"$2\" "
                   "--output_path=%s/\"$1\"_%s_\"$2\"\n"
                   % (config.DATA_DIR, config.MODEL_DIRS, params, config.ROLLOUT_DIRS,params))
    f_script.close()
    os.system("chmod +x %s" % path2)

    # Rendering script
    path3 = "%s_rendering.sh" % bash_dir
    f_script = open(path3, "w")
    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -ne 3 ];then\n")
    f_script.write("    echo \"Need 3 argument: data_name test_id rollout_ofsset\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")
    f_script.write("python -m learning_to_simulate.render_rollout "
                   "--rollout_path=%s/\"$1\"_%s_\"$2\"/rollout_test_0_$3.pkl "
                   "--out_path=%s/\"$1\"/_%s_$2 --test_id=$2 --rollout_offset=$3\n" % (config.ROLLOUT_DIRS,params, config.DATA_DIR,params))

    f_script.close()
    os.system("chmod +x %s" % path3)





def copy_data_files(data_name_path=None):
    folders1 = glob.glob("%s/*/" % config.ORIGIN_DATA_DIR)
    f_data = open(data_name_path, "w")
    for folder in folders1:
        print("Folder: ", folder)
        folders2 = glob.glob("%s/*_raw/" % folder)
        dt = folder.split("/")[-2].lower()
        for folder2 in folders2:
            tracking_path = folder2.replace("_raw", "_tracking")
            cmd = "cp %s*.csv \"%s\"" % (tracking_path, folder2)
            print(cmd)
            os.system(cmd)

            dName = folder2.split("/")[-2].split("(")[0].lower().split("_")[0]
            dName = "%s_%s" % (dt, dName)
            print("Dname: ", dName)

            target_dir_path = "%s/%s/" % (config.DATA_DIR, dName)
            print("Target: ", target_dir_path)
            utils.ensuredir(target_dir_path)
            cmd = "cp %s*.csv \"%s\"" % (folder2, target_dir_path)
            cmd = cmd.replace("(", "\(").replace(")", "\)")
            print(cmd)

            os.system(cmd)
            f_data.write("%s\n" % dName)

            # xfolder2 = folder2.replace("(トラッキング)", "(元画像)")
            # parts = folder2.split("/")
            # dnam2 = parts[-2]
            # sep_id = dnam2.index("_")
            # # partern = "%s_%s*元画像*" % (dnam2[:sep_id], dnam2[sep_id+1:sep_id+4])
            # parts[-2] = partern
            # xfolder2 = "/".join(["/".join(parts[:-2]), partern, "/".join(parts[-1:])])
            cmd = "cp %s*.avi \"%s\"" % (folder2, target_dir_path)

            #  cmd = cmd.replace("(", "\(").replace(")", "\)")

            print(cmd)
            os.system(cmd)
    f_data.close()


if __name__ == "__main__":
    copy_data_files("%s/data_names.txt" % config.C_DIR)

    create_bash_script("%s/run_1" % config.C_DIR)
