# -*- coding: utf-8 -*-

import sys
import os

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..", "T-GCN", "T-GCN-TensorFlow")
)

sys.path.append(base_dir)   
import pickle as pkl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_metr_data
from tgcn import tgcnCell
from gru import GRUCell 

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

time_start = time.time()

###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 5, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 100, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 7, 'time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 64, 'batch size.')
flags.DEFINE_string('dataset', 'metr', 'metr dataset')
flags.DEFINE_string('model_name', 'TGCN_att','TGCN_att')
### spatial-attention settings (our addition; use_spatial_att=False reproduces the baseline) ###
flags.DEFINE_boolean('use_spatial_att', True, 'enable the spatial attention block.')
flags.DEFINE_integer('sat_units', 16, 'projection dim d_a of the spatial attention keys/queries.')
flags.DEFINE_boolean('sat_mask_adj', True, 'restrict spatial attention to the static adjacency support.')
flags.DEFINE_float('sat_gamma_init', 0.0, 'init of the residual gate; 0.0 => starts identical to baseline.')
flags.DEFINE_integer('sat_vis_batches', 20, 'number of test batches averaged for the spatial attention heatmap.')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
use_spatial_att = FLAGS.use_spatial_att
sat_units = FLAGS.sat_units
sat_mask_adj = FLAGS.sat_mask_adj
sat_gamma_init = FLAGS.sat_gamma_init
sat_vis_batches = FLAGS.sat_vis_batches


###### load data ######
data,adj = load_metr_data()

print("Data:",data.shape)
print("Adj:",adj.shape)

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)



### Perturbation Analysis
#noise = np.random.normal(0,0.2,size=data.shape)
#noise = np.random.poisson(16,size=data.shape)
#scaler = MinMaxScaler()
#scaler.fit(noise)
#noise = scaler.transform(noise)
#data1 = data1 + noise



#### normalization
max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

print("trainX:", trainX.shape)
print("trainY:", trainY.shape)
print("testX:", testX.shape)
print("testY:", testY.shape)

totalbatch = math.ceil(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

###### static adjacency support mask (our addition) ######
# adj is the SAME predefined matrix already consumed by tgcnCell; it is neither modified
# nor replaced here. We only derive a binary support mask from it so that spatial
# attention stays inside the graph neighbourhood instead of becoming a dense learned graph.
adj_mask_np = (np.asarray(adj) > 0).astype(np.float32)
adj_mask = tf.constant(adj_mask_np, dtype=tf.float32, name='adj_support_mask')
print("Adj support: %.1f%% of node pairs, mean degree %.1f"
      % (100.0 * adj_mask_np.mean(), adj_mask_np.sum(1).mean()))

def spatial_attention(x, weight_sat, bias_sat):
    """Spatial (node-wise) attention -- OUR ADDITION.

    Runs on the raw model input, BEFORE graph convolution + GRU, and learns how strongly
    every source node j should contribute to the representation of every target node i.
    It attends over the NODE axis; the time axis is used only as the per-node descriptor,
    so this is spatial attention, not temporal attention.

        x   : [batch, seq_len, num_nodes]     (the tensor the baseline feeds to tgcnCell)
        out : [batch, seq_len, num_nodes]     (drop-in replacement, shape unchanged)
        s   : [batch, num_nodes, num_nodes]   (row i = attention target node i pays to sources)
    """
    # [B, T, N] -> [B, N, T]. Node-major layout: row i is node i's length-T speed profile
    # over the input window, which is the feature vector used to compare nodes.
    x_n = tf.transpose(x, perm=[0, 2, 1])

    # Collapse batch and node axes so one shared dense layer applies to every node of every
    # sample: [B, N, T] -> [B*N, T]. This is a batched dense layer, not a shape hack -- the
    # same W is used for all nodes, which is what makes the scores comparable across nodes.
    flat = tf.reshape(x_n, shape=[-1, seq_len])

    # Query / key projections:  Q = X W_q + b_q,  K = X W_k + b_k
    q = tf.matmul(flat, weight_sat['wq']) + bias_sat['bq']      # [B*N, d_a]
    k = tf.matmul(flat, weight_sat['wk']) + bias_sat['bk']      # [B*N, d_a]
    q = tf.reshape(q, shape=[-1, num_nodes, sat_units])         # [B, N, d_a]
    k = tf.reshape(k, shape=[-1, num_nodes, sat_units])         # [B, N, d_a]

    # Pairwise compatibility  E_ij = <q_i, k_j> / sqrt(d_a)     -> [B, N, N]
    e = tf.matmul(q, k, transpose_b=True)
    e = e / tf.sqrt(tf.cast(sat_units, tf.float32))

    # Structural prior: keep the predefined adjacency as a hard support constraint, so a
    # node can only attend to nodes it is actually connected to. The adjacency itself is
    # untouched -- this only masks the attention logits (-1e9 -> 0 after softmax).
    if sat_mask_adj:
        e = e * adj_mask + (adj_mask - 1.0) * 1e9

    # Row-wise softmax over the SOURCE-node axis:  sum_j s[b,i,j] = 1
    s = tf.nn.softmax(e, axis=-1)                               # [B, N, N]

    # Attention aggregation: context[b,i,:] = sum_j s[b,i,j] * x_n[b,j,:]
    context = tf.matmul(s, x_n)                                 # [B, N, T]

    # Gated residual. gamma is trainable and initialised to sat_gamma_init (0.0 by default),
    # so at initialisation the block is the identity and the model is exactly the original
    # A3T-GCN; it only injects spatial context if that reduces the loss.
    out = x_n + weight_sat['gamma'] * context                   # [B, N, T]

    # Back to the layout the rest of the (unmodified) pipeline expects.
    out = tf.transpose(out, perm=[0, 2, 1])                     # [B, T, N]
    return out, s


def TGCN(_X, weights, biases, weight_sat=None, bias_sat=None):
    ### spatial attention (our addition) -- on the input, before graph convolution + GRU
    if use_spatial_att:
        _X, spatial_alpha = spatial_attention(_X, weight_sat, bias_sat)
    else:
        spatial_alpha = tf.constant(0.0, name='spatial_alpha_disabled')

    ### existing TGCN (graph convolution) + GRU -- unchanged
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[seq_len,-1,num_nodes,gru_units])
    out = tf.transpose(out, perm=[1,0,2,3])

    ### existing temporal attention -- unchanged
    last_output,alpha = self_attention1(out, weight_att, bias_att)

    ### existing output layer -- unchanged
    output = tf.reshape(last_output,shape=[-1,seq_len])
    output = tf.matmul(output, weights['out']) + biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])

    return output, outputs, states, alpha, spatial_alpha
    
def self_attention1(x, weight_att,bias_att):
    x = tf.matmul(tf.reshape(x,[-1,gru_units]),weight_att['w1']) + bias_att['b1']
#    f = tf.layers.conv2d(x, ch // 8, kernel_size=1, kernel_initializer=tf.variance_scaling_initializer())
    f = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    g = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    h = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']

    f1 = tf.reshape(f, [-1,seq_len])
    g1 = tf.reshape(g, [-1,seq_len])
    h1 = tf.reshape(h, [-1,seq_len])
    s = g1 * f1
    print('s',s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print('bata',beta)
    context = tf.expand_dims(beta,2) * tf.reshape(x,[-1,seq_len,num_nodes])

    context = tf.transpose(context,perm=[0,2,1])
    print('context', context)
    return context, beta 
 

###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# weights
weights = {
    'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight_o')}
bias = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}
weight_att={
    'w1':tf.Variable(tf.random_normal([gru_units,1], stddev=0.1),name='att_w1'),
    'w2':tf.Variable(tf.random_normal([num_nodes,1], stddev=0.1),name='att_w2')}
bias_att = {
    'b1': tf.Variable(tf.random_normal([1]),name='att_b1'),
    'b2': tf.Variable(tf.random_normal([1]),name='att_b2')}
# spatial attention parameters (our addition). Declared AFTER the existing variables so the
# baseline variables keep their original creation order, and only when the block is enabled
# so that use_spatial_att=False leaves the original model (and its L2 term) untouched.
if use_spatial_att:
    _glorot = tf.compat.v1.keras.initializers.glorot_uniform()
    weight_sat = {
        'wq': tf.Variable(_glorot([seq_len, sat_units]), name='sat_wq'),
        'wk': tf.Variable(_glorot([seq_len, sat_units]), name='sat_wk'),
        'gamma': tf.Variable(sat_gamma_init, dtype=tf.float32, name='sat_gamma')}
    bias_sat = {
        'bq': tf.Variable(tf.zeros([sat_units]), name='sat_bq'),
        'bk': tf.Variable(tf.zeros([sat_units]), name='sat_bk')}
else:
    weight_sat, bias_sat = None, None

if model_name == 'TGCN_att':
    pred, ttto, ttts, alpha, spatial_alpha = TGCN(inputs, weights, bias, weight_sat, bias_sat)
    
y_pred = pred
#print('ooooo',y_pred)
             

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))

lr_var = tf.Variable(lr, trainable=False, dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(lr_var).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

#out = 'out/%s'%(model_name)
out = 'out1withdyn/%s'%(model_name)
# tag keeps the two arms of the comparison in separate folders; the baseline
# (use_spatial_att=False) keeps exactly the original path.
sa_tag = '_spatialatt' if use_spatial_att else ''
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r%s'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch,sa_tag)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 
    return batch_s
    
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]

#dynamic learning rate

best_rmse = float('inf')
bad_epochs = 0
patience = 7
factor = 0.5
min_lr = 1e-5
  
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output, alpha1 = sess.run([optimizer, loss, error, y_pred, alpha],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)

    # check whether rmse improved

    if rmse < best_rmse:
        best_rmse = rmse
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        current_lr = sess.run(lr_var)
        new_lr = max(
            current_lr * factor,
            min_lr
        )

        sess.run(
            tf.assign(
                lr_var,
                new_lr
            )
        )

        print(
            "Learning rate reduced:",
            current_lr,
            "->",
            new_lr
        )

        bad_epochs = 0

    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)

    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))
    
    if (epoch % 500 == 0):        
        saver.save(sess, path+'/model_100/graphGRU_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
#x = [i for i in range(training_epoch)]
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
#test_rmse = [float(i) for i in test_rmse]
var = pd.DataFrame(batch_loss1)
var.to_csv(path+'/batch_loss.csv',index = False,header = False)
var = pd.DataFrame(train_loss)
var.to_csv(path+'/train_loss.csv',index = False,header = False)
var = pd.DataFrame(batch_rmse1)
var.to_csv(path+'/batch_rmse.csv',index = False,header = False)
var = pd.DataFrame(train_rmse)
var.to_csv(path+'/train_rmse.csv',index = False,header = False)
var = pd.DataFrame(test_loss)
var.to_csv(path+'/test_loss.csv',index = False,header = False)
var = pd.DataFrame(test_acc)
var.to_csv(path+'/test_acc.csv',index = False,header = False)
var = pd.DataFrame(test_rmse)
var.to_csv(path+'/test_rmse.csv',index = False,header = False)


index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv',index = False,header = False)
plot_result(test_result,test_label1,path)
plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,test_r2,path)

fig1 = plt.figure(figsize=(7,3))
ax1 = fig1.add_subplot(1,1,1)
print(alpha1.shape)
plt.plot(np.sum(alpha1,0))
plt.savefig(path+'/alpha.jpg',dpi=500)
plt.show()


fig1 = plt.figure(figsize=(8,2))

attention = np.sum(alpha1, axis=0)

plt.imshow(np.mat(attention),
           aspect='auto',
           cmap='viridis')

plt.xticks(np.arange(attention.shape[0]))

plt.yticks([])

plt.xlabel("Historical Time Steps")
plt.title("Temporal Attention Heatmap")
plt.colorbar(label="Attention Weight")

plt.tight_layout()
plt.savefig(path + '/alpha11.jpg', dpi=500, bbox_inches='tight')
plt.show()

############## spatial attention visualization (our addition) ###############
# The spatial attention matrix is input-dependent: softmax is computed per SAMPLE, giving
# S of shape [batch, num_nodes, num_nodes]. Materialising it for the whole test set would
# need len(testX)*207*207*4 bytes (~1.2 GB), so we stream a fixed number of test batches
# and accumulate the MEAN over samples. The reported matrix is therefore
#     S_mean[i, j] = E_over_test_samples[ attention target node i pays to source node j ]
# The node axes are NOT averaged away -- only the sample axis is.
if use_spatial_att:
    n_vis = min(sat_vis_batches * batch_size, testX.shape[0])
    spatial_sum = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    n_seen = 0
    for m in range(0, n_vis, batch_size):
        vb = testX[m:m + batch_size]
        sa_b = sess.run(spatial_alpha, feed_dict={inputs: vb})
        spatial_sum += sa_b.sum(axis=0)
        n_seen += sa_b.shape[0]
    spatial_alpha_mean = (spatial_sum / n_seen).astype(np.float32)   # [num_nodes, num_nodes]

    print('spatial_alpha (per batch) shape:', sa_b.shape)
    print('spatial_alpha_mean shape:', spatial_alpha_mean.shape,
          '| averaged over', n_seen, 'test samples')
    print('learned residual gate gamma =', sess.run(weight_sat['gamma']))

    # raw matrix for later analysis
    pd.DataFrame(spatial_alpha_mean).to_csv(
        path + '/spatial_alpha_matrix.csv', index=False, header=False)

    # --- full N x N heatmap: x = source nodes, y = target nodes ---
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(spatial_alpha_mean, aspect='auto', cmap='viridis')
    plt.xlabel("Source node j (influencing node)")
    plt.ylabel("Target node i (predicted node)")
    plt.title("Spatial Attention Heatmap  S[i, j]\n(mean over %d test samples)" % n_seen)
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(path + '/spatial_alpha_heatmap.jpg', dpi=500, bbox_inches='tight')
    plt.show()

    # --- zoomed 30x30 block: readable node indices ---
    k = min(30, num_nodes)
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(spatial_alpha_mean[:k, :k], aspect='auto', cmap='viridis')
    plt.xticks(np.arange(k), np.arange(k), rotation=90, fontsize=6)
    plt.yticks(np.arange(k), np.arange(k), fontsize=6)
    plt.xlabel("Source node j")
    plt.ylabel("Target node i")
    plt.title("Spatial Attention (first %d nodes)" % k)
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(path + '/spatial_alpha_zoom.jpg', dpi=500, bbox_inches='tight')
    plt.show()

    # --- global node importance: column sum = total attention a node RECEIVES as a source ---
    node_influence = spatial_alpha_mean.sum(axis=0)                  # [num_nodes]
    pd.DataFrame(node_influence).to_csv(
        path + '/spatial_node_influence.csv', index=False, header=False)
    top = np.argsort(-node_influence)[:20]
    fig = plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(top)), node_influence[top], color='steelblue')
    plt.xticks(np.arange(len(top)), top, rotation=90, fontsize=7)
    plt.xlabel("Node index")
    plt.ylabel("Total attention received")
    plt.title("Top-20 most influential nodes (column sums of S)")
    plt.tight_layout()
    plt.savefig(path + '/spatial_node_influence.jpg', dpi=500, bbox_inches='tight')
    plt.show()

    print('top-10 influential nodes:', top[:10].tolist())

############## run summary (our addition, for the baseline-vs-spatial comparison) ###############
summary = pd.DataFrame([{
    'use_spatial_att': use_spatial_att,
    'sat_units': sat_units if use_spatial_att else None,
    'sat_mask_adj': sat_mask_adj if use_spatial_att else None,
    'gamma_final': float(sess.run(weight_sat['gamma'])) if use_spatial_att else None,
    'lr': lr, 'epochs': training_epoch, 'batch_size': batch_size,
    'gru_units': gru_units, 'seq_len': seq_len, 'pre_len': pre_len,
    'best_epoch': index,
    'rmse': float(np.min(test_rmse)), 'mae': float(test_mae[index]),
    'accuracy': float(test_acc[index]), 'r2': float(test_r2[index]),
    'var': float(test_var[index]),
    'train_seconds': float(time_end - time_start),
}])
summary.to_csv(path + '/run_summary.csv', index=False)

print(
    'min_rmse:%r' % np.min(test_rmse),
    'latest_rmse:%r' % test_rmse[-1],
    'min_mae:%r' % test_mae[index],
    'max_acc:%r' % test_acc[index],
    'r2:%r' % test_r2[index],
    'var:%r' % test_var[index]
)