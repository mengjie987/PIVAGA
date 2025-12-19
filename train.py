from __future__ import division
from __future__ import print_function
from VGAE.evaluation import compute_scores,compute_scores2
from VGAE.input_data import load_data
from VGAE.model import *
from VGAE.optimizer import OptimizerAE, OptimizerVAE
from VGAE.preprocessing import *
import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn_ae', 'Name of the model')
''' Available Models:

- gcn_ae: Graph Autoencoder from Kipf and Welling (2016), with 2-layer
          GCN encoder and inner product decoder

- gcn_vae: Variational Graph Autoencoder from Kipf and Welling (2016), with
           Gaussian priors, 2-layer GCN encoders and inner product decoder
'''

# Model parameters
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
# flags.DEFINE_integer('epochs', 1000, 'Number of epochs in training.')
#修改1  1000到10，方便计算
flags.DEFINE_integer('epochs',30, 'Number of epochs in training.')
# flags.DEFINE_integer('epochs',1000, 'Number of epochs in training.')
flags.DEFINE_boolean('features', False, 'Include node features or not in GCN')
flags.DEFINE_float('lamb', 1., 'lambda parameter from Gravity AE/VAE models \
                                as introduced in section 3.5 of paper, to \
                                balance mass and proximity terms')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 512, 'Number of units in GCN hidden layer.')
flags.DEFINE_integer('dimension', 256, 'Dimension of GCN output: \
- equal to embedding dimension for standard AE/VAE and source-target AE/VAE')

# Experimental setup parameters
flags.DEFINE_integer('nb_run', 1, 'Number of model run + test')
flags.DEFINE_float('prop_val', 5., 'Proportion of edges in validation set \
                                   (for Task 1)')
flags.DEFINE_float('prop_test', 10., 'Proportion of edges in test set \
                                      (for Tasks 1 and 2)')
flags.DEFINE_boolean('validation', False, 'Whether to report validation \
                                           results  at each epoch (for \
                                           Task 1)')
flags.DEFINE_boolean('verbose', True, 'Whether to print comments details.')

# Lists to collect average results
mean_roc = []
mean_ap = []
mean_ac = []
mean_re = []
mean_f1 = []
mean_time = []
mean_aupr = []
mean_precision = []

# Load graph dataset
if FLAGS.verbose:
    print("Loading data...")
adj_init, features = load_data()
#修改7 确定学习率
#首先定义了横坐标和总坐标数组
# x = []#用于存放横坐标
# t_loss = []#用于存放train_loss
# v_loss = []#用于存放train_loss
roc_tu = []#用于存放train_loss
# data_dict = {}
mm = ['0','1','2','3','4','5']
# The entire training process is repeated FLAGS.nb_run times
for i in range(FLAGS.nb_run):

    # Edge Masking: compute Train/Validation/Test set
    if FLAGS.verbose:
        print("Masking test edges...")
    
    # Edge masking for General Directed Link Prediction
    adj, val_edges, val_edges_false, test_edges, test_edges_false = \
        DTI_prediction(adj_init, FLAGS.prop_test,
                                                FLAGS.prop_val)
    # Preprocessing and initialization
    if FLAGS.verbose:
        print("Preprocessing and Initializing...")
    # Compute number of nodes
    num_nodes = adj.shape[0]
    # If features are not used, replace feature matrix by identity matrix
    if not FLAGS.features:
        features = sp.identity(adj.shape[0])
    # Preprocessing on node features
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = None
    if FLAGS.model == 'gcn_ae':
        # Standard Graph Autoencoder
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif FLAGS.model == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(placeholders, num_features, num_nodes,
                            features_nonzero)
    else:
        raise ValueError('Undefined model!')

    # Optimizer (see tkipf/gae original GAE repository for details)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]
                                                - adj.sum()) * 2)
    with tf.name_scope('optimizer'):
        # Optimizer for Non-Variational Autoencoders
        if FLAGS.model in ('gcn_ae'):
            opt = OptimizerAE(preds = model.reconstructions,
                              labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices = False), [-1]),
                              pos_weight = pos_weight,
                              norm = norm)
        # Optimizer for Variational Autoencoders
        elif FLAGS.model in ('gcn_vae'):
            opt = OptimizerVAE(preds = model.reconstructions,
                               labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                             validate_indices = False), [-1]),
                               model = model,
                               num_nodes = num_nodes,
                               pos_weight = pos_weight,
                               norm = norm)

    # Normalization and preprocessing on adjacency matrix
    adj_norm = preprocess_graph(adj)
    adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

    # Initialize TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model training
    if FLAGS.verbose:
        print("Training...")
    # Flag to compute total running time
    t_start = time.time()
    for epoch in range(FLAGS.epochs):
        # Flag to compute running time for each epoch
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                        placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        if FLAGS.verbose:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t),end = ' ')
            # Validation (implemented for Task 1 only)
            #if FLAGS.validation and FLAGS.task == 'task_1':
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict = feed_dict)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # val_roc, val_ap = compute_scores(val_edges, val_edges_false, emb)
            val_roc, val_ap, aupr,val_ac, val_re_score,val_f1,val_pre = compute_scores(val_edges, val_edges_false, emb)

            print("val_ROC=", "{:.5f}".format(val_roc), "val_AP=", "{:.5f}".format(val_ap),"AUPR=", "{:.5f}".format(aupr),"precision=", "{:.5f}".format(val_pre),"accuracy=", "{:.5f}".format(val_ac), "recall=", "{:.5f}".format(val_re_score),"F1=", "{:.5f}".format(val_f1))
            # val_roc, val_ap = compute_scores(val_edges, val_edges_false, emb)

            # print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

            roc_tu.append("{:.5f}".format(val_roc))

    str = '\n'
    # Compute total running time
    mean_time.append(time.time() - t_start)

    # Get embedding from model

    #修改 这一行便于计算AUC和AP
    # emb = sess.run(model.z_mean, feed_dict = feed_dict)
    emb = sess.run(model.re, feed_dict = feed_dict)
    np.savetxt('results_score.txt', emb, delimiter=',')


    # Test model
    if FLAGS.verbose:
        print("Testing model...")
    # Compute ROC and AP scores on test sets
    roc_score, ap_score ,aupr,ac_score, re_score,f1_s,pre_s= compute_scores(test_edges, test_edges_false, emb)
    results_predict, results_label = compute_scores2(test_edges, test_edges_false, emb)

    np.savetxt("results_predict"+ mm[i] +".txt", results_predict, delimiter=',')
    np.savetxt("results_label"+ mm[i] +".txt", results_label, delimiter=',')
    # roc_score, ap_score= compute_scores(test_edges, test_edges_false, emb)
    # Append to list of scores over all runs
    mean_roc.append(roc_score)
    mean_ap.append(ap_score)
    mean_ac.append(ac_score)
    mean_re.append(re_score)
    mean_f1.append(f1_s)
    mean_aupr.append(aupr)
    mean_precision.append(pre_s)

# Report final results
print("\nTest results for", FLAGS.model,"\n",
      "___________________________________________________\n")

print("AUC:            ", mean_roc, "\t","Mean:",np.mean(mean_roc),"\t","Std:",np.std(mean_roc))
print("AP:             ", mean_ap,"\t","Mean:",np.mean(mean_ap),"\t","Std:",np.std(mean_ap))
print("AUPR:           ", mean_aupr,"\t","Mean:",np.mean(mean_aupr),"\t","Std:",np.std(mean_aupr))
print("Recall:         ", mean_re,"\t","Mean:",np.mean(mean_re),"\t","Std:",np.std(mean_re))
print("precision:      ", mean_precision,"\t","Mean:",np.mean(mean_precision),"\t","Std:",np.std(mean_precision))
print("F1:             ", mean_f1,"\t","Mean:",np.mean(mean_f1),"\t","Std:",np.std(mean_f1))
print("accuracy:       ", mean_ac,"\t","Mean:",np.mean(mean_ac),"\t","Std:",np.std(mean_ac))
print("Running times:  ", mean_time,"\t","Mean:",np.mean(mean_time),"\t","Std:",np.std(mean_time))

