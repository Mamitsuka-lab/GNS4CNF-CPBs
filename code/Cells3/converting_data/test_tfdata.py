import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_sequence_record(qid, relevances, features):
    context_feature = {"qid": _int64_feature(qid)}

    feature_list = dict()
    feature_list["relevance"] = tf.train.FeatureList(
        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=[x])) for x in relevances]
    )
    feature_list["feature"] = tf.train.FeatureList(
        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=feature)) for feature in features]
    )

    context = tf.train.Features(feature=context_feature)
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    print(example)
    exit(-1)
    # return example.SerializeToString()
    return example


def demo():
    qid = 1000
    relevances =[np.random.uniform() for _ in range(2)]
    features = [[np.random.uniform() for _ in range(2)] for _ in range(100)]
    example = make_sequence_record(qid, relevances, features)
    data = example.SerializeToString()
    fout = open("/tmp/example.tt", "wb")
    fout.write(data)
    fout.close()

def testLoad():

    tfdata = tf.data.TFRecordDataset(["/tmp/example.tt"])
    print(tfdata)

if __name__ == "__main__":
    # demo()
    print(11 // 4 )
    # testLoad()
