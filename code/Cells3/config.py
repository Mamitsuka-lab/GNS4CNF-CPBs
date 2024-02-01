import os

C_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = "%s/data3" % C_DIR

TEST1_DIR = "%s/hHigh" % DATA_DIR
TEST2_DIR = "%s/ControlHX" % DATA_DIR
ORIGIN_DATA_DIR = "%s/../CellData/" % C_DIR
MODEL_DIRS = "/tmp/models3"
ROLLOUT_DIRS = "/tmp/rollouts3"
DATA_PREX = "hmsc"
TYPE_2_INT = {"control": 0, "low": 1, "high": 2}
ENVS = [0, 0.001, 0.02]
FILE_TEST_IDS = [0, 4, 8]
TYPE_SIZE = 4
CLEAN = False
SQRT = True
# N_POINT_DIS = 4
N_POINT_DIS = 4
C = 6  # 2, 3, 6
TEST_ID = 0
N_PN = 10
OFF_SET = C
TEST_ALL = True
