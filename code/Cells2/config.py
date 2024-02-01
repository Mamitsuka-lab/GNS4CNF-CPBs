import os

C_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = "%s/data2" % C_DIR

O_DATA = ""

TEST1_DIR = "%s/hHigh" % DATA_DIR
TEST2_DIR = "%s/ControlHX" % DATA_DIR
ORIGIN_DATA_DIR = "%s/../CellData/" % C_DIR
MODEL_DIRS = "/tmp/models"
ROLLOUT_DIRS = "/tmp/rollouts"
CLEAN = False
SQRT = True
# N_POINT_DIS = 4
N_POINT_DIS = 4
C = 6 # 2, 3, 6
TEST_ID = 0
N_PN = 10
OFF_SET = C
ROLLOUT_OFFSET = 10
TEST_ALL = True