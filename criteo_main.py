# coding=UTF-8
import pandas as pd
import tensorflow as tf
from deepfm import model_fn
from criteo_data_load import input_fn


def get_hparams():
    vocab_sizes = {
      'gender': 3, 'age': 7, 'tagid': 312089, 'province': 35,
        'city': 316
    }
    # step = tf.Variable(0, trainable=False)
    # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     [10000, 15000], [1e-0, 1e-1, 1e-2])
    # # lr and wd can be a function or a tensor
    # lr = 1e-3 * schedule(step)
    # wd = lambda: 1e-4 * schedule(step)
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam'
    )

    return {
        'embed_dim': 256,
        'vocab_sizes': vocab_sizes,
        'multi_embed_combiner': 'sum',
        # 在这个case中，没有多个field共享同一个vocab的情况，而且field_name和vocab_name相同
        'field_vocab_mapping': {'gender': 'gender', 'age': 'age', 'tagid': 'tagid', 'province': 'province',
                                'city': 'city'}, #, 'make':'make', 'model':'model'},
        'dropout_rate': 0.1,
        'batch_norm': False,
        'hidden_units': [128, 64],
        'optimizer': optimizer
    }


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(999)
    hparams = get_hparams()
    deepfm = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir='models/criteo',
                                    params=hparams)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_file='/Users/gl.j/PycharmProjects/baseline/data/训练集/new2_train.csv',
                                           n_repeat=5,
                                           batch_size=64,
                                           batches_per_shuffle=10), max_steps=50000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_file='/Users/gl.j/PycharmProjects/baseline/data/val.csv',
                                           n_repeat=1,
                                           batch_size=64,
                                           batches_per_shuffle=-1))

    tf.estimator.train_and_evaluate(deepfm, train_spec, eval_spec)

    pred_dict = deepfm.predict(input_fn=lambda: input_fn(data_file='/Users/gl.j/PycharmProjects/baseline/0_100000',
                                                         n_repeat=1,
                                                         batch_size=64,
                                                         batches_per_shuffle=-1))
    print(pred_dict)
    res = []
    for pred_res in pred_dict:
        res.append(pred_res['probabilities'])
    res = pd.DataFrame(res)
    res.to_csv('res.csv')

    test = pd.read_csv('/Users/gl.j/PycharmProjects/baseline/data/测试集/test.txt', sep=',', header=None)
    test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model']
    test1 = pd.read_csv('/Users/gl.j/PycharmProjects/baseline/res.csv', sep=',')
    test1.columns = ['user_id', 'score']
    test1['user_id'] = test['pid']
    test1['category_id'] = 1
    test1['rank'] = test1['score'].rank()
    test1.loc[test1['rank'] <= int(test1.shape[0] * 0.5), 'category_id'] = 0
    # test1['category_id'] = test1['category_id'].apply(lambda x: 1 if x >= 0.5 else 0)
    test1[['user_id', 'category_id']].to_csv('new_res.csv', index=False)
