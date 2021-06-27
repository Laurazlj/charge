import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf 
import global_config
import model
import random
from prec_recall_counter import PrecRecallCounter
from init import *
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import sys
import pandas as pd
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


reload(sys)
sys.setdefaultencoding("utf-8")
def ismixed(a, b, c):
	#mixed couple, pred, answer
	
	temp = False
	for i in c:
		temp = temp|(i == a[0])
	return temp&(b == a[1])&(b not in c)
def create_dir(ds):
	for temp in ds:
		if not os.path.exists(temp):
			os.makedirs(temp)
def main(_):

	path = os.getcwd()
	father_path = os.path.dirname(path)
	checkpoint_dir = 	father_path + "/checkpoint/"
	lstm_log_dir = 		father_path + "/log/evaluation_charge_log/"
	attr_log_dir = 		father_path + "/log/evaluation_attr_log/"
	val_lstm_log_dir = 	father_path + "/log/validation_charge_log/"
	val_attr_log_dir = 	father_path + "/log/validation_attr_log/"

	create_dir([checkpoint_dir,lstm_log_dir,attr_log_dir,val_lstm_log_dir,val_attr_log_dir])
	restore  = False
	skiptrain = False
	valmatrix = False
	val_case = False
	mixandmatrix = False
	single_attr_log = False
	bs = 64
	perstep = 500
	eva_number = 0
	val_number = 0
	mixcouple = [69,71]
	mixattr = [2,3,7]
	single_attr = [4,9]
	
	word2id,word_embeddings,attr_table,x_train,y_train,y_attr_train,x_test,y_test,y_attr_test,x_val,y_val,y_attr_val,namehash,length_train,length_test,length_val = load_data_and_labels_fewshot()
	id2word = {}
	for i in word2id:
		id2word[word2id[i]] = i
	batches = batch_iter(list(zip(x_train, y_train, y_attr_train)), global_config.batch_size, global_config.num_epochs)
	lstm_config = model.lstm_Config()
	lstm_config.num_steps = len(x_train[0])
	lstm_config.hidden_size = len(word_embeddings[0])
	lstm_config.vocab_size = len(word_embeddings)
	lstm_config.num_classes = len(y_train[0])
	lstm_config.num_epochs = 20
	lstm_config.batch_size = bs

	lstm_eval_config = model.lstm_Config()
	lstm_eval_config.keep_prob = 1.0
	lstm_eval_config.num_steps = len(x_train[0])
	lstm_eval_config.hidden_size = len(word_embeddings[0])
	lstm_eval_config.vocab_size = len(word_embeddings)
	lstm_eval_config.num_classes = len(y_train[0])
	lstm_eval_config.batch_size = bs
	lstm_eval_config.num_epochs = 20

	zero_x = [0 for i in range(lstm_config.num_steps)]
	zero_y = [0 for i in range(lstm_config.num_classes)]

	lstm_count_tab = np.array([[0.0 for i in range(lstm_config.num_classes)]for j in range(lstm_config.num_classes)])
	total_tab = np.array([0.0 for i in range(lstm_config.num_classes)])	
	with tf.Graph().as_default():
		tf.set_random_seed(6324)
		tf_config = tf.ConfigProto() 
		tf_config.gpu_options.allow_growth = True 
		sess = tf.Session(config=tf_config) 
		with sess.as_default():
			lstm_initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("lstm_model", reuse=None, initializer = lstm_initializer):
				placeholders = {
					'support': tf.placeholder(tf.float32, shape=(None, None)),
					'labels': tf.placeholder(tf.float32, shape=(None, lstm_config.num_classes)),
					'labels_mask': tf.placeholder(tf.int32),
					'dropout': tf.placeholder_with_default(0., shape=()),
					'num_features_nonzero': tf.placeholder(tf.int32),
				}
				print 'lstm step1'
				lstm_model = model.LSTM_MODEL(placeholders=placeholders,word_embeddings=word_embeddings,attr_table=attr_table,config = lstm_config)
				print 'lstm step2'
				lstm_optimizer = tf.train.AdamOptimizer(lstm_config.lr)
				print 'lstm step3'
				lstm_global_step = tf.Variable(0, name = "lstm_global_step", trainable = False)
				lstm_train_op = lstm_optimizer.minimize(lstm_model.total_loss,global_step = lstm_global_step)
				print 'lstm step4'
			saver = tf.train.Saver()
			init_op = tf.initialize_all_variables()
			sess.run(init_op)
			best_macro_f1 = 0.0
			loss_list = [[]for i in range(4)]
			valid_loss = [[]for i in range(4)]
			if restore:
				f_f1 = open(val_lstm_log_dir+'best_macro_f1','r')
				f1s = f_f1.readlines()
				best_macro_f1 = float(f1s[-1].strip().split(' ')[-1].strip('[').strip(']'))
				f_f1.close()
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
				if ckpt and ckpt.model_checkpoint_path:  
					saver.restore(sess, ckpt.model_checkpoint_path)  
				else:  
					pass

			def lstm_train_step(x_batch, y_batch, y_attr_batch, length_batch, support, features, y_train, train_mask):
				"""
				A single training step
				"""
				feed_dict = {
					lstm_model.input_x: x_batch,
					lstm_model.input_length: length_batch,
					lstm_model.input_y: y_batch,
					lstm_model.unmapped_input_attr: y_attr_batch,
					lstm_model.keep_prob: 1.0,
					lstm_model.gcn_inputs: features,
				}
				feed_dict.update({placeholders['labels']: y_train})
				feed_dict.update({placeholders['labels_mask']: train_mask})
				feed_dict.update({placeholders['support']: support})
				feed_dict.update({placeholders['num_features_nonzero']: 0})
				feed_dict.update({placeholders['dropout']: 0.0})
				_, step, total_loss, lstm_loss, attr_loss, gcn_loss = sess.run(
					[lstm_train_op, lstm_global_step, lstm_model.total_loss, lstm_model.lstm_loss, lstm_model.total_attr_loss, lstm_model.gcn_loss], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				if(step == 1):
					loss_list[0].append(total_loss)
					loss_list[1].append(lstm_loss)
					loss_list[2].append(attr_loss)
					loss_list[3].append(gcn_loss)
				if step % 50 == 0:
					#print sc
	 				print("{}: step {}, total loss {:g}, lstm_loss {:g}, attr_loss {:g}, gcn_loss {:g}".format(time_str, step, total_loss,
	 					lstm_loss, attr_loss, gcn_loss))
				if step % 200 == 0:
					loss_list[0].append(total_loss)
					loss_list[1].append(lstm_loss)
					loss_list[2].append(attr_loss)
					loss_list[3].append(gcn_loss)
				return step
			def lstm_dev_step(x_batch, y_batch, y_attr_batch, length_batch, support, features, y_train, train_mask, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
					lstm_model.input_x: x_batch,
					lstm_model.input_length: length_batch,
					lstm_model.input_y: y_batch,
					lstm_model.unmapped_input_attr: y_attr_batch,
					lstm_model.keep_prob: 1.0,
					lstm_model.gcn_inputs: features,
				}
				feed_dict.update({placeholders['labels']: y_train})
				feed_dict.update({placeholders['labels_mask']: train_mask})
				feed_dict.update({placeholders['support']: support})
				feed_dict.update({placeholders['num_features_nonzero']: 0})
				feed_dict.update({placeholders['dropout']: 0.0})
				runlist = [lstm_model.predictions,lstm_model.attr_preds,lstm_model.total_loss,lstm_model.lstm_loss,lstm_model.total_attr_loss,lstm_model.attn_weights,lstm_model.gcn_loss]
				
				lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss = sess.run(runlist, feed_dict=feed_dict)
				
				return lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss

			batches = batch_iter(list(zip(x_train,y_train,y_attr_train,length_train)),lstm_config.batch_size, lstm_config.num_epochs)

			num_per_epoch = (int)(round(len(x_train)/lstm_config.batch_size)) 
			step = 0
			x_1 = [0]
			cost_val = []
			for batch in batches:

				if(step < 2000):
					lstm_config.lr = 0.02
				elif(step >= 2000 & step < 5000):
					lstm_config.lr = 0.01
				elif(step >= 5000 & step < 10000):
					lstm_config.lr = 1e-3
				else:
					lstm_config.lr = 1e-4
				
				if ((step % perstep) == 0) or (skiptrain):
					

					print 'Evaluation'
					if mixandmatrix:
						f_mix = open(lstm_log_dir+str(eva_number)+'mixed.html','w')
						f_mix.write('<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/></head>\n')
					if single_attr_log:
						f_single_attr = open(lstm_log_dir+str(eva_number)+'attr.html','w')
						f_single_attr.write('<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/></head>\n')

					
					if(step != 0):
						x = [0]
						cnt = step/200
						for i in range(cnt):
							number = (i+1)*200
							x.append(number)
						y_0 = loss_list[0]
						y_1 = loss_list[1]
						y_2 = loss_list[2]
						y_3 = loss_list[3]
						fig = plt.figure(figsize = (7,5))
						p1 = pl.plot(x, y_0,'g-',label=u'total loss')
						pl.legend()
						p2 = pl.plot(x, y_1,'r-', label = u'lstm loss')
						pl.legend()
						p3 = pl.plot(x, y_2, 'b-', label = u'attr loss')
						pl.legend()
						p4 = pl.plot(x, y_3, 'y-', label = u'gcn loss')
						pl.legend()
						pl.xlabel(u'step')
						pl.ylabel(u'loss')
						plt.title('loss in training')
						plt.savefig('./train_loss.png')
						plt.close()
					
					
					all_count = 0.0
					total_losses,lstm_losses,attr_losses,gcn_losses = 0.0,0.0,0.0,0.0
					lstm_prc = PrecRecallCounter(lstm_config.num_classes,lstm_log_dir,'lstm',eva_number)
					attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],attr_log_dir,'attr',eva_number)
					lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
					num = int(len(y_test)/float(lstm_eval_config.batch_size))
					print num
					for i in range(num):
						if i %100 == 0:
							print i
						begin = i * lstm_eval_config.batch_size
						end = (i+1) * lstm_eval_config.batch_size
						y_batch_t = y_test[begin:end]
						x_batch_t = x_test[begin:end]
						y_attr_batch_t = y_attr_test[begin:end]
						length_batch = length_test[begin:end]
						
						if(eva_number == 0):
							adj, features, y_data, mask = build_graph(y_batch_t, x_batch_t, word_embeddings)
							features = preprocess_features(features)
							support = preprocess_adj(adj)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/features"+ str(i) +".npy",features)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/support"+ str(i) +".npy",support)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/y_data"+ str(i) +".npy",y_data)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/mask"+ str(i) +".npy",mask)
						else:
							support = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/support"+ str(i) +".npy")
							features = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/features"+ str(i) +".npy")
							y_data = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/y_data"+ str(i) +".npy")
							mask = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/mask"+ str(i) +".npy")
						
						lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch,support,features,y_data,mask)
						total_losses+=t_loss
						lstm_losses+=l_loss
						attr_losses+=a_loss
						gcn_losses+=g_loss
						for j in range(lstm_eval_config.batch_size):
							indexes = np.flatnonzero(y_batch_t[j]) # location of nonzero elements
							lstm_prc.multicount(lstm_p[j], indexes)
							for index in indexes:
								lstm_matrix[index][lstm_p[j]] += 1
							for k in range(global_config.num_of_attr):
								attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
							if mixandmatrix:
								mixed = ismixed(mixcouple, lstm_p[j], indexes)
								if mixed:
									wordcolor = '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n'
									f_mix.write('<p>'+str(lstm_p[j])+' '+str(indexes)+'</p>\n')
									towrite = ''
									for k in range(global_config.num_of_attr):
										towrite = towrite + str(attr_p[j][k]) + ' '
									f_mix.write('<p>'+towrite+'</p>\n')
									towrite = ''
									for k in range(global_config.num_of_attr):
										towrite = towrite + str(y_attr_batch_t[j][k]) + ' '
									f_mix.write('<p>'+towrite+'</p>\n')
									for c in mixattr:
										f_mix.write(wordcolor%(0,str(c)))
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											f_mix.write(wordcolor%(attn_weights[j][c][w]/np.max(attn_weights[j][c]),id2word[x_batch_t[j][w]]))
										f_mix.write('<p>---</p>\n')
							if single_attr_log:
								for attr_index in single_attr:
									if (attr_p[j][attr_index] != y_attr_batch_t[j][attr_index])&(y_attr_batch_t[j][attr_index]!=2):
										wordcolor = '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n'
										f_single_attr.write('<p>'+str(indexes)+str(attr_index)+' '+str(attr_p[j][attr_index])+' '+str(y_attr_batch_t[j][attr_index])+'</p>\n')
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											f_single_attr.write(wordcolor%(attn_weights[j][attr_index][w]/np.max(attn_weights[j][attr_index]),id2word[x_batch_t[j][w]]))
										f_single_attr.write('<p>---</p>\n')

					begin = num * lstm_eval_config.batch_size
					y_batch_t = y_test[begin:]
					x_batch_t = x_test[begin:]
					y_attr_batch_t = y_attr_test[begin:]
					length_batch = length_test[begin:]
					cl = len(y_batch_t)
					for itemp in range(lstm_eval_config.batch_size-cl):
						y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
						x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
						y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
						length_batch = np.append(length_batch,[length_batch[0]],axis=0)

					if(eva_number == 0):
						adj, features, y_data, mask = build_graph(y_batch_t, x_batch_t, word_embeddings)
						features = preprocess_features(features)
						support = preprocess_adj(adj)
						np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/features"+ str(num) +".npy",features)
						np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/support"+ str(num) +".npy",support)
						np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/y_data"+ str(num) +".npy",y_data)
						np.save("/mnt/d/jjpeng/ljzhao/charge2/test_graph/mask"+ str(num) +".npy",mask)
					else:
						support = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/support"+ str(num) +".npy")
						features = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/features"+ str(num) +".npy")
						y_data = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/y_data"+ str(num) +".npy")
						mask = np.load("/mnt/d/jjpeng/ljzhao/charge2/test_graph/mask"+ str(num) +".npy")
					
					lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch,support,features,y_data,mask)
					total_losses+=t_loss
					lstm_losses+=l_loss
					attr_losses+=a_loss
					gcn_losses+=g_loss
					for jtemp in range(cl):
						indexes = np.flatnonzero(y_batch_t[jtemp])
						lstm_prc.multicount(lstm_p[jtemp], indexes)
						for index in indexes:
							lstm_matrix[index][lstm_p[jtemp]] += 1
						for k in range(global_config.num_of_attr):
							attr_prc.count(attr_p[jtemp][k], y_attr_batch_t[jtemp][k], k)
							
					lstm_prc.compute()
					attr_prc.compute()

					lstm_prc.output()
					attr_prc.output()

					if (lstm_prc.macro_f1[0] > best_macro_f1) or skiptrain:
						best_macro_f1 = lstm_prc.macro_f1[0]
						f_f1 = open(val_lstm_log_dir+'best_macro_f1','a+')
						f_f1.write('eva:'+str(eva_number)+' '+str(best_macro_f1)+'\n')
						f_f1.close()
					
						print 'Validation'
						if not skiptrain:
							saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
						all_count = 0.0
						total_losses,lstm_losses,attr_losses,gcn_losses = 0.0,0.0,0.0,0.0
						val_lstm_prc = PrecRecallCounter(lstm_config.num_classes,val_lstm_log_dir,'lstm',val_number)
						val_attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],val_attr_log_dir,'attr',val_number)
						val_lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
						num = int(len(y_val)/float(lstm_eval_config.batch_size))
						if val_case:
							f_case = open(val_lstm_log_dir+'case'+str(val_number),'w')
						print num
						for i in range(num):
							if i %100 == 0:
								print i
							begin = i * lstm_eval_config.batch_size
							end = (i+1) * lstm_eval_config.batch_size
							y_batch_t = y_val[begin:end]
							x_batch_t = x_val[begin:end]
							y_attr_batch_t = y_attr_val[begin:end]
							length_batch = length_val[begin:end]
							
							if(val_number == 0):
								adj, features, y_data, mask = build_graph(y_batch_t, x_batch_t, word_embeddings)
								features = preprocess_features(features)
								support = preprocess_adj(adj)
								np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/features"+ str(i) +".npy",features)
								np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/support"+ str(i) +".npy",support)
								np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/y_data"+ str(i) +".npy",y_data)
								np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/mask"+ str(i) +".npy",mask)
							else:
								support = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/support"+ str(i) +".npy")
								features = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/features"+ str(i) +".npy")
								y_data = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/y_data"+ str(i) +".npy")
								mask = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/mask"+ str(i) +".npy")
							lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch,support,features,y_data,mask)
							total_losses+=t_loss
							lstm_losses+=l_loss
							attr_losses+=a_loss
							gcn_losses+=g_loss
							for j in range(lstm_eval_config.batch_size):
								indexes = np.flatnonzero(y_batch_t[j])
								val_lstm_prc.multicount(lstm_p[j], indexes)
								for index in indexes:
									val_lstm_matrix[index][lstm_p[j]] += 1
								for k in range(global_config.num_of_attr):
									val_attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
								if val_case:
									towrite = str(lstm_p[j])+'\t'+str(indexes[0])+'\t'+str(attr_p[j])+'\t'+str(y_attr_batch_t[j])+'\t'
									for w in range(len(x_batch_t[j])):
										if w == length_batch[j]:
											break
										towrite = towrite + id2word[x_batch_t[j][w]]+' '
									for temp_attr in range(global_config.num_of_attr):
										towrite = towrite + '\t'
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											towrite = towrite + str(attn_weights[j][temp_attr][w]/np.max(attn_weights[j][temp_attr]))+' '
									towrite = towrite + '\n'
									f_case.write(towrite)
						begin = num * lstm_eval_config.batch_size
						y_batch_t = y_val[begin:]
						x_batch_t = x_val[begin:]
						y_attr_batch_t = y_attr_val[begin:]
						length_batch = length_val[begin:]
						cl = len(y_batch_t)
						for itemp in range(lstm_eval_config.batch_size-cl):
							y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
							x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
							y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
							length_batch = np.append(length_batch,[length_batch[0]],axis=0)
									
						if(val_number == 0):
							adj, features, y_data, mask = build_graph(y_batch_t, x_batch_t, word_embeddings)
							features = preprocess_features(features)
							support = preprocess_adj(adj)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/features"+ str(num) +".npy",features)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/support"+ str(num) +".npy",support)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/y_data"+ str(num) +".npy",y_data)
							np.save("/mnt/d/jjpeng/ljzhao/charge2/val_graph/mask"+ str(num) +".npy",mask)
						else:
							support = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/support"+ str(num) +".npy")
							features = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/features"+ str(num) +".npy")
							y_data = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/y_data"+ str(num) +".npy")
							mask = np.load("/mnt/d/jjpeng/ljzhao/charge2/val_graph/mask"+ str(num) +".npy")
						lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights,g_loss = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch,support,features,y_data,mask)
						total_losses+=t_loss
						lstm_losses+=l_loss
						attr_losses+=a_loss
						gcn_losses+=g_loss
						valid_loss[0].append(total_losses/num)
						valid_loss[1].append(lstm_losses/num)
						valid_loss[2].append(attr_losses/num)
						valid_loss[3].append(gcn_losses/num)
						cost_val.append(total_losses/num)
						
						if(step != 0):
							x_0 = [0]
							cnt = step/200
							for i in range(cnt):
								number = (i+1)*200
								x_0.append(number)
							y_0 = loss_list[0]
							x_1.append(step)
							y_1 = valid_loss[0]
							fig = plt.figure(figsize = (7,5))
							pl.plot(x_0, y_0,'g-',label=u'training loss')
							pl.legend()
							p2 = pl.plot(x_1, y_1, 'b-', label = u'validation loss')
							pl.legend()
							pl.xlabel(u'step')
							pl.ylabel(u'loss')
							plt.title('loss in training and validation')
							plt.savefig('./compare.png')
							plt.close()
						
						for jtemp in range(cl):
							indexes = np.flatnonzero(y_batch_t[jtemp])
							val_lstm_prc.multicount(lstm_p[jtemp], indexes)
							for index in indexes:
								val_lstm_matrix[index][lstm_p[jtemp]] += 1
							for k in range(global_config.num_of_attr):
								val_attr_prc.count(attr_p[jtemp][k], y_attr_batch_t[jtemp][k], k)
							if val_case:
									towrite = str(lstm_p[jtemp])+'\t'+str(indexes[0])+'\t'+str(attr_p[jtemp])+'\t'+str(y_attr_batch_t[jtemp])+'\t'
									for w in range(len(x_batch_t[jtemp])):
										if w == length_batch[jtemp]:
											break
										towrite = towrite + id2word[x_batch_t[jtemp][w]]+' '
									for temp_attr in range(global_config.num_of_attr):
										towrite = towrite + '\t'
										for w in range(len(x_batch_t[jtemp])):
											if w == length_batch[jtemp]:
												break
											towrite = towrite + str(attn_weights[jtemp][temp_attr][w]/np.max(attn_weights[jtemp][temp_attr]))+' '
									towrite = towrite + '\n'
									f_case.write(towrite)

						val_lstm_prc.compute()
						val_attr_prc.compute()
						val_lstm_prc.output()
						val_attr_prc.output()
						if valmatrix:
							fm = open(val_lstm_log_dir+str(val_number)+'matrix','w')
							for i in range(lstm_config.num_classes):
								towrite = ""
								for j in range(lstm_config.num_classes):
									towrite = towrite + str(val_lstm_matrix[i][j])+' '
								towrite = towrite + '\n'
								fm.write(towrite)
							fm.close()
						val_number += 1


					if mixandmatrix:
						fm = open(lstm_log_dir+str(eva_number)+'matrix','w')
						for i in range(lstm_config.num_classes):
							towrite = ""
							for j in range(lstm_config.num_classes):
								towrite = towrite + str(lstm_matrix[i][j])+' '
							towrite = towrite + '\n'
							fm.write(towrite)
						fm.close()

					num = float(num)
					tn = datetime.datetime.now()
					print tn.isoformat()
					print 'loss total:{:g}, lstm:{:g}, attr:{:g}, gcn:{:g}'.format(total_losses/num,lstm_losses/num,attr_losses/num,gcn_losses/num)
					if skiptrain:
						break
					eva_number += 1
					if len(cost_val) < 10:
						avg = np.mean(cost_val)
					else:
						avg = np.mean(cost_val[-11:-1])
					if step/num_per_epoch > 8 and cost_val[-1] > avg:
						break

				x_batch, y_batch, y_attr_batch, length_batch = zip(*batch)
				adj, features, y_data, mask = build_graph(y_batch, x_batch, word_embeddings)
				features = preprocess_features(features)
				support = preprocess_adj(adj)
				step = lstm_train_step(x_batch, y_batch, y_attr_batch, length_batch, support, features, y_data, mask)
						


def build_graph(doc_name_list, doc_content_list, word_embeddings):
	word_embeddings_dim = 200
	shuffle_doc_words_list = list(doc_content_list)
	doc_name_list = list(doc_name_list)

	# build vocab
	word_freq = {}
	word_set = set()
	for doc_words in shuffle_doc_words_list:
		for word in doc_words:
			if word in word_freq:
				word_freq[word] += 1
			else:
				word_freq[word] = 1
				word_set.add(word)

	vocab = list(word_set)
	vocab_size = len(vocab)

	word_doc_list = {}

	for i in range(len(shuffle_doc_words_list)):
		doc_words = shuffle_doc_words_list[i]
		words = doc_words
		appeared = set()
		for word in words:
			if word in appeared:
				continue
			if word in word_doc_list:
				doc_list = word_doc_list[word]
				doc_list.append(i)
				word_doc_list[word] = doc_list
			else:
				word_doc_list[word] = [i]
			appeared.add(word)


	word_doc_freq = {}
	for word, doc_list in word_doc_list.items():
		word_doc_freq[word] = len(doc_list)


	word_id_map = {}
	for i in range(vocab_size):
		word_id_map[vocab[i]] = i
	
	# x: feature vectors of training docs, no initial features
	# slect 90% training set
	train_size = len(doc_name_list)

	y = np.array(doc_name_list)

	# allx: the the feature vectors of both labeled and unlabeled training instances
	# (a superset of x)
	# unlabeled training instances -> words

	word_vectors = np.random.uniform(-0.01, 0.01,
									(vocab_size, 100))

	for i in range(len(vocab)):
		word = vocab[i]
		word_vectors[i] = word_embeddings[int(word)]

	row_allx = []
	col_allx = []
	data_allx = []

	for i in range(vocab_size):
		for j in range(word_embeddings_dim):
			row_allx.append(int(i))
			col_allx.append(j)
			if(j <= 99):
				data_allx.append(word_vectors.item((i, j)))
			else:
				data_allx.append(0)
			


	row_allx = np.array(row_allx)
	col_allx = np.array(col_allx)
	data_allx = np.array(data_allx)

	allx = sp.csr_matrix(
		(data_allx, (row_allx, col_allx)), shape=(vocab_size, word_embeddings_dim))

	ally = doc_name_list

	for i in range(vocab_size):
		one_hot = [0 for l in range(149)]
		ally.append(one_hot)

	ally = np.array(ally)

	'''
	Doc word heterogeneous graph
	'''

	# word co-occurence with context windows
	window_size = 20
	windows = []

	for doc_words in shuffle_doc_words_list:
		words = doc_words
		length = len(words)
		if length <= window_size:
			windows.append(words)
		else:
			for j in range(length - window_size + 1):
				window = words[j: j + window_size]
				windows.append(window)


	word_window_freq = {}
	for window in windows:
		appeared = set()
		for i in range(len(window)):
			if window[i] in appeared:
				continue
			if window[i] in word_window_freq:
				word_window_freq[window[i]] += 1
			else:
				word_window_freq[window[i]] = 1
			appeared.add(window[i])

	word_pair_count = {}
	for window in windows:
		for i in range(1, len(window)):
			for j in range(0, i):
				word_i = window[i]
				word_i_id = word_id_map[word_i]
				word_j = window[j]
				word_j_id = word_id_map[word_j]
				if word_i_id == word_j_id:
					continue
				word_pair_str = str(word_i_id) + ',' + str(word_j_id)
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] += 1
				else:
					word_pair_count[word_pair_str] = 1
				# two orders
				word_pair_str = str(word_j_id) + ',' + str(word_i_id)
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] += 1
				else:
					word_pair_count[word_pair_str] = 1

	row = []
	col = []
	weight = []

	# pmi as weights

	num_window = len(windows)

	for key in word_pair_count:
		temp = key.split(',')
		i = int(temp[0])
		j = int(temp[1])
		count = word_pair_count[key]
		word_freq_i = word_window_freq[vocab[i]]
		word_freq_j = word_window_freq[vocab[j]]
		pmi = log((1.0 * count / num_window) /
				(1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
		if pmi <= 0:
			continue
		row.append(train_size + i)
		col.append(train_size + j)
		weight.append(pmi)

	# doc word frequency
	doc_word_freq = {}

	for doc_id in range(len(shuffle_doc_words_list)):
		doc_words = shuffle_doc_words_list[doc_id]
		words = doc_words
		for word in words:
			word_id = word_id_map[word]
			doc_word_str = str(doc_id) + ',' + str(word_id)
			if doc_word_str in doc_word_freq:
				doc_word_freq[doc_word_str] += 1
			else:
				doc_word_freq[doc_word_str] = 1

	for i in range(len(shuffle_doc_words_list)):
		doc_words = shuffle_doc_words_list[i]
		words = doc_words
		doc_word_set = set()
		for word in words:
			if word in doc_word_set:
				continue
			j = word_id_map[word]
			key = str(i) + ',' + str(j)
			freq = doc_word_freq[key]
			if i < train_size:
				row.append(i)
			else:
				row.append(i + vocab_size)
			col.append(train_size + j)
			idf = log(1.0 * len(shuffle_doc_words_list) /
					word_doc_freq[vocab[j]])
			weight.append(freq * idf)
			doc_word_set.add(word)

	node_size = train_size + vocab_size
	adj = sp.csr_matrix(
		(weight, (row, col)), shape=(node_size, node_size))

	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	features = allx.tolil()
	labels = ally
	idx_train = range(len(y))
	train_mask = sample_mask(idx_train, labels.shape[0])
	y_train = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	
	return adj, features, y_train, train_mask

if __name__ == "__main__":
	tf.app.run()








