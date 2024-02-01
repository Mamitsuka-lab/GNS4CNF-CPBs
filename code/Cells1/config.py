import os

C_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = "%s/datax" % C_DIR

TEST1_DIR = "%s/hHigh" % DATA_DIR
TEST2_DIR = "%s/ControlHX" % DATA_DIR
ORIGIN_DATA_DIR = "/home/gpux1/Downloads/CellData"
MODEL_DIRS = "/tmp/modelsx"
ROLLOUT_DIRS = "/tmp/rolloutsx"
CLEAN = False

# N_POINT_DIS = 4
N_POINT_DIS = 4
C = 6  # 2, 3, 6
N_TEST2 = 5

TEST_ID = 1
OFF_SET = 20
TEST_ALL = True
SQRT = True

N_PN = 10