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
    f_script.write("if [ $# -ne 1 ];then\n")
    f_script.write("    echo \"Need 1 argument: data_name\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")
    f_script.write("python -m learning_to_simulate.train --data_path=%s/$1 --model_path=%s/\"$1\"_%s\n"
                   % (config.DATA_DIR, config.MODEL_DIRS, params))
    f_script.close()
    os.system("chmod +x %s" % path1)

    # Rollout script
    path2 = "%s_rollout.sh" % bash_dir
    f_script = open(path2, "w")
    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -gt 2 ];then\n")
    f_script.write("    echo \"Need 1-2 argument: data_name test_id{optional}\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("v1=$1\nv2=${2:0}\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")

    f_script.write("python -m learning_to_simulate.train --mode=\"eval_rollout\" --data_path=%s/$v1 "
                   "--model_path=%s/\"$v1\"_%s "
                   "--output_path=%s/\"$v1\"_%s  --test_id=$v2\n"
                   % (config.DATA_DIR, config.MODEL_DIRS, params, config.ROLLOUT_DIRS,params))
    f_script.close()
    os.system("chmod +x %s" % path2)

    # Rendering script
    path3 = "%s_rendering.sh" % bash_dir
    f_script = open(path3, "w")
    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -gt 3 ];then\n")
    f_script.write("    echo \"Need 2-3 argument: data_name rendering_model{test|train}\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("v1=$1\nv2=$2\nv3=${3:0}\n")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")
    f_script.write("python -m learning_to_simulate.render_rollout "
                   "--rollout_path=%s/\"$1\"_%s/rollout_test_0_$v3.pkl "
                   "--out_path=%s/\"$1\"/$2_%s --test_id=$v3\n" % (config.ROLLOUT_DIRS,params, config.DATA_DIR,params))

    f_script.close()
    os.system("chmod +x %s" % path3)

    # Run all scripts:
    pathx = "%s_all.sh" % bash_dir
    f_script = open(pathx, "w")

    f_script.write("#!/bin/bash\n")
    f_script.write("if [ $# -gt 3 ];then\n")
    f_script.write("    echo \"Need 2-3 argument: data_name rendering_model{test|train} test_id{optional}\"\n")
    f_script.write("    exit -1\n")
    f_script.write("fi\n")
    f_script.write("v3=${3:0}")
    f_script.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
    f_script.write("conda activate pycell\n")
    f_script.write("%s $1\n" % path1)
    f_script.write("%s $1 $v3\n" % path2)
    f_script.write("%s $1 $2 $v3\n" % path3)

    f_script.close()
    os.system("chmod +x %s" % pathx)


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
    from utilities import ensuredir
    print("Start...")
    ensuredir(config.DATA_DIR)
    copy_data_files("%s/data_names.txt" % config.C_DIR)
    create_bash_script("%s/run_1" % config.C_DIR)
