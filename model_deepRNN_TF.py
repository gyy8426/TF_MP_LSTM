import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
from cocoeval import COCOScorer
import unicodedata
import copy
gpu_id = 0

class Video_Caption_MP_RNN():
    
    def __init__(self, options):
        self.options = options
        #with tf.device('/cpu:0'):
        self.initialiser = tf.random_uniform_initializer(-self.options['init_scale'], self.options['init_scale'])

        self.Wemb = tf.Variable(tf.random_uniform([self.options['num_words'], self.options['dim_word']], -self.options['init_scale'], self.options['init_scale']), name='Wemb')
        self.lstm_sen = tf.contrib.rnn.LSTMCell(self.options['dim_lstm_sen'], use_peepholes = True, state_is_tuple = True)
        self.lstm_sen_drop = tf.contrib.rnn.DropoutWrapper(self.lstm_sen, output_keep_prob= 1 - self.options['drop_out_rate'])
        self.lstm_feat_sen = tf.contrib.rnn.LSTMCell(self.options['dim_lstm_feat_sen'], use_peepholes = True, state_is_tuple = True)
        self.lstm_feat_sen_drop = tf.contrib.rnn.DropoutWrapper(self.lstm_feat_sen, output_keep_prob= 1 - self.options['drop_out_rate'])
        self.logit_word_W = tf.Variable(tf.random_uniform([self.options['dim_lstm_feat_sen'], self.options['num_words']], -self.options['init_scale'],self.options['init_scale']), name='logit_word_W')
        self.logit_word_b = tf.Variable(tf.zeros([self.options['num_words']]), name='logit_word_b')

        
    def build_model(self):
        #feat shape: batch_size, num_frames, dim_feat
        #sen shape: batch_size, len_sen
         #self.options['batch_size']
        #self.batch_size = tf.placeholder(tf.int32, [1])
        self.maxlen = self.options['max_len_sen']
        feat = tf.placeholder(tf.float32, [None, self.options['num_frames'],self.options['dim_feat']])
        sen = tf.placeholder(tf.int32, [None, self.maxlen])
        feat_mask = tf.placeholder(tf.float32, [None, self.options['num_frames']])
        sen_mask = tf.placeholder(tf.float32, [None, self.maxlen])
        self.batch_size = tf.shape(feat)[0]
        
        '''
        init_sen = tf.placeholder(tf.float32, [self.batch_size, self.options['dim_lstm_sen']])
        cur_sen_state = (init_sen, init_sen)
        init_feat_sen = tf.placeholder(tf.float32, [self.batch_size, self.options['dim_lstm_feat_sen']])
        cur_feat_sen_state = (init_feat_sen, init_feat_sen)
        cur_emb = tf.placeholder([self.batch_size,self.options['dim_word']])
        '''
        c_init_sen = tf.zeros([self.batch_size, self.options['dim_lstm_sen']])
        m_init_sen = tf.zeros([self.batch_size, self.options['dim_lstm_sen']])
        cur_sen_state = (c_init_sen, m_init_sen)
        c_init_feat_sen = tf.zeros([self.batch_size, self.options['dim_lstm_feat_sen']])
        m_init_feat_sen = tf.zeros([self.batch_size, self.options['dim_lstm_feat_sen']])
        cur_feat_sen_state = (c_init_feat_sen, c_init_feat_sen)        
        cur_emb = tf.zeros([self.batch_size,self.options['dim_word']])
        cur_cost = 0.0
        feat_mean_pooling = tf.reduce_mean(feat, axis=1)
        #shape: batch_size, dim_feat
        pro_word_list = []
        with tf.variable_scope('model') as scope: 
            for word_i in range(self.maxlen):
                if word_i > 0: scope.reuse_variables()
                with tf.variable_scope("lstm_sen"):
                    lstm_sen_output, next_sen_state = self.lstm_sen_drop(cur_emb, cur_sen_state)
                #lstm_sen_output shape: batch_size, dim_lstm_sen
                with tf.variable_scope("lstm_feat_sen"):                
                    lstm_feat_sen_output, next_feat_sen_state = self.lstm_feat_sen_drop(tf.concat([lstm_sen_output,feat_mean_pooling],axis = -1), cur_feat_sen_state)
                #lstm_feat_sen_output shape: batch_size, dim_lstm_feat_sen
                logit_word = tf.nn.xw_plus_b( lstm_feat_sen_output, self.logit_word_W, self.logit_word_b)
                #logit_word shape: batch_size, num_words
                #logit_word = tf.reshape(logit_word, [logit_word.shape[0],-1])
                pdf_word = tf.nn.softmax(logit_word,dim=-1)
                #pdf_word shape: batch_size, num_words
                label_word = sen[:,word_i] # b 
                label_word_1 = tf.expand_dims(label_word,1)
                batch_id = tf.range(self.batch_size)# b
                batch_id_1 = tf.expand_dims(batch_id,1)
                batch_id_word = tf.concat([batch_id_1, label_word_1], 1) 
                pro_word = tf.gather_nd(pdf_word,batch_id_word) #find the pdf_word[batch_id][label_word]
                pro_word_list.append(pro_word)
                # shape: batch_size
                pro_log = -tf.log(pro_word + 1e-8)
                cost_mask = pro_log * sen_mask[:,word_i]
                cur_emb = tf.nn.embedding_lookup(self.Wemb,label_word)
                cur_cost = cur_cost + tf.reduce_mean(cost_mask)
                cur_sen_state = next_sen_state
                cur_feat_sen_state = next_feat_sen_state
                # get cost over the first dim
        cost = cur_cost
        loss = cost
        model_weight = []
        if self.options['decay_c'] > 0.0:
            decay_c = tf.constant(np.float32(self.options['decay_c']), name = 'decay_c')
            weight_decay = 0.
            for vv in tf.trainable_variables():
                print 'trainable_variables:',vv
                model_weight.append(vv)
                weight_decay += tf.reduce_sum((vv ** 2))
            weight_decay *= decay_c
            loss += weight_decay
        f_train_input = {'feat':feat, 'feat_mask':feat_mask, 'sen':sen, 'sen_mask':sen_mask}

                       #  'init_sen':init_sen, 'init_feat_sen':init_feat_sen,'cur_emb':cur_emb} 
        f_train_output = {'pro_word':pro_word_list, 
                           'cost':cost, 'loss':loss}
        return f_train_input, f_train_output,model_weight
        
        
    def build_sample_model(self):
        #generate sample with beam search
        #feat shape: batch_size, num_frames, dim_feat
        #sen shape: batch_size, len_sen 
        feat = tf.placeholder(tf.float32, [None,self.options['num_frames'],self.options['dim_feat']])
        feat_mask = tf.placeholder(tf.float32, [None,self.options['num_frames']])
        feat_mask = tf.cast(feat_mask, tf.float32)
        feat_mean_pooling = tf.reduce_mean(feat, axis=-2)
        c_init_sen = tf.zeros([ self.options['dim_lstm_sen']])
        m_init_sen = tf.zeros([ self.options['dim_lstm_sen']])
        init_sen_state = (c_init_sen, m_init_sen)
        
        c_init_feat_sen = tf.zeros([ self.options['dim_lstm_feat_sen']])
        m_init_feat_sen = tf.zeros([ self.options['dim_lstm_feat_sen']])
        init_feat_sen_state = (c_init_feat_sen, c_init_feat_sen) 
        
        f_init_inputs = {'feat': feat, 'feat_mask' :feat_mask}
        f_init_outputs = {'feat':feat, 'c_init_sen': c_init_sen, 'm_init_sen': m_init_sen,\
        'c_init_feat_sen':c_init_feat_sen,'m_init_feat_sen':m_init_feat_sen}
        
        c_cur_sen = tf.placeholder(tf.float32, [None, self.options['dim_lstm_sen']])
        m_cur_sen = tf.placeholder(tf.float32, [None, self.options['dim_lstm_sen']])
        cur_sen_state = (c_cur_sen,m_cur_sen)
        
        c_cur_feat_sen = tf.placeholder(tf.float32, [None, self.options['dim_lstm_feat_sen']])
        m_cur_feat_sen = tf.placeholder(tf.float32, [None, self.options['dim_lstm_feat_sen']])
        cur_feat_sen_state = (c_cur_feat_sen,m_cur_feat_sen)
        cur_word = tf.placeholder(tf.int32, [None])
        
        init_emb = tf.placeholder(tf.float32, [None, self.options['dim_word']])
        
        with tf.variable_scope('model') as scope: 
            scope.reuse_variables()
            cur_emb = tf.where(tf.less(cur_word,0),init_emb, tf.nn.embedding_lookup(self.Wemb,cur_word))
            #shape: batch_size, dim_feat
            
            with tf.variable_scope("lstm_sen"):
                lstm_sen_output, next_sen_state = self.lstm_sen(cur_emb, cur_sen_state)
            #lstm_sen_output shape: batch_size, dim_lstm_sen
            with tf.variable_scope("lstm_feat_sen"):                
                lstm_feat_sen_output, next_feat_sen_state = self.lstm_feat_sen(tf.concat([lstm_sen_output,feat_mean_pooling],axis=-1), cur_feat_sen_state)
            #lstm_feat_sen_output shape: batch_size, dim_lstm_feat_sen
            logit_word = tf.nn.xw_plus_b( lstm_feat_sen_output, self.logit_word_W, self.logit_word_b)
            #logit_word shape: batch_size, num_words
            #logit_word = tf.reshape(logit_word, [logit_word.shape[0],-1])
            pdf_word = tf.nn.softmax(logit_word,dim = -1)
            next_word = tf.argmax(pdf_word,axis=1)
        c_next_sen = next_sen_state[0]
        m_next_sen = next_sen_state[1]
        c_cur_next_sen = next_feat_sen_state[0]
        m_cur_next_sen = next_feat_sen_state[1]        
        f_next_input = {'feat': feat, 'feat_mask' :feat_mask,'cur_word':cur_word,'init_emb':init_emb,\
                        'c_cur_sen':c_cur_sen,'m_cur_sen':m_cur_sen,\
                        'c_cur_feat_sen':c_cur_feat_sen,'m_cur_feat_sen':m_cur_feat_sen}
                        
        f_next_output = {'pdf_word': pdf_word, 'next_word' :next_word,\
                        'c_next_sen':c_next_sen,'m_next_sen':m_next_sen,\
                        'c_cur_next_sen':c_cur_next_sen,'m_cur_next_sen':m_cur_next_sen}
        return  f_init_inputs, f_init_outputs, f_next_input, f_next_output
        
    def gen_sen_beam_search(self, sess, feat, feat_mask, 
                            f_init_inputs,f_init_outputs,f_next_input,f_next_output,                              
                            trng=None,
                            k=1, maxlen=30, stochastic=False):
        
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')

        # [(26,1024),(512,),(512,)]
        rval = sess.run([f_init_outputs['feat'], \
                         f_init_outputs['c_init_sen'],f_init_outputs['m_init_sen'],\
                         f_init_outputs['c_init_feat_sen'],f_init_outputs['m_init_feat_sen']], \
                         feed_dict={f_init_inputs['feat']:feat[None,:],f_init_inputs['feat_mask']:feat_mask[None,:]})
        ctx0 = rval[0]

        # next lstm and stacked lstm state and memory
        next_states = []
        next_memorys = []
        n_layers_lstm = 2
        for lidx in xrange(n_layers_lstm):
            next_states.append([])
            next_memorys.append([])
            next_states[lidx].append(rval[2*lidx+1])
            next_states[lidx][-1] = next_states[lidx][-1].reshape([live_k, next_states[lidx][-1].shape[0]])
            next_memorys[lidx].append(rval[2*lidx+2])
            next_memorys[lidx][-1] = next_memorys[lidx][-1].reshape([live_k, next_memorys[lidx][-1].shape[0]])

        next_w = -1 * np.ones((1,)).astype('int32')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            live_num = next_states[0][0].shape[0]
            init_emb = np.zeros(shape=[live_num,self.options['dim_word']]).astype('float32')
            feat_temp = np.repeat(feat[None,:],live_num,axis=0)
            feat_mask_temp = np.repeat(feat_mask[None,:],live_num,axis=0)
            rval = sess.run([f_next_output['pdf_word'], f_next_output['next_word'],\
                             f_next_output['c_next_sen'],f_next_output['m_next_sen'],\
                             f_next_output['c_cur_next_sen'],f_next_output['m_cur_next_sen']], \
                         feed_dict={f_next_input['feat']:feat_temp,f_next_input['feat_mask']:feat_mask_temp,\
                                    f_next_input['cur_word']:next_w,f_next_input['init_emb']:init_emb,\
                                    f_next_input['c_cur_sen']:next_states[0][0],f_next_input['m_cur_sen']:next_memorys[0][0],\
                                    f_next_input['c_cur_feat_sen']:next_states[1][0],f_next_input['m_cur_feat_sen']:next_memorys[1][0]})
            next_p = rval[0]
            next_w = rval[1] # already argmax sorted

            next_states = []
            next_memorys = []
            for lidx in xrange(n_layers_lstm):
                next_states.append([])
                next_memorys.append([])
                next_states[lidx].append(rval[2*lidx+2])
                next_memorys[lidx].append(rval[2*lidx+3])

            if stochastic:
                sample.append(next_w[0]) # take the most likely one
                sample_score += next_p[0,next_w[0]]
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,50000)
                cand_scores = hyp_scores[:,None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size # index of row
                word_indices = ranks_flat % voc_size # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')

                new_hyp_states = []
                new_hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_states.append([])
                    new_hyp_memories.append([])
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])   # before is ti
                    for lidx in np.arange(n_layers_lstm):
                        new_hyp_states[lidx].append(copy.copy(next_states[lidx][0][ti]))
                        new_hyp_memories[lidx].append(copy.copy(next_memorys[lidx][0][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = [] 
                hyp_states = []
                hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    hyp_states.append([])
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = np.array([w[-1] for w in hyp_samples])
                next_states = []
                next_memorys = []
                for lidx in xrange(n_layers_lstm):
                    next_states.append([])
                    next_memorys.append([])
                    next_states[lidx].append(np.array(hyp_states[lidx]))
                    next_memorys[lidx].append(np.array(hyp_memories[lidx]))

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score, next_states, next_memorys
            
            
            
            
            
            
            
            