import tensorflow as tf
import sys
import time
from layers import Layers
import data_engine
import metrics
from optimizers import *
from predict import *
from model_deepRNN_TF import Video_Caption_MP_RNN
import numpy as np
def validate_options(options):
    if options['ctx2out']:
        warnings.warn('Feeding context to output directly seems to hurt.')
    if options['dim_word'] > options['dim_lstm_feat_sen']:
        warnings.warn('dim_word should only be as large as dim_lstm_feat_sen.')
    return options


class Attention(object):
    def __init__(self, channel=None):
        self.rng_numpy, self.rng_theano = get_two_rngs()
        self.layers = Layers()
        self.predict = Predict()
        self.channel = channel

    def load_params(self, path, params):
        # load params from disk
        pp = np.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive'%kk)
            params[kk] = pp[kk]

        return params

    def pred_probs(self, whichset, options, sess, f_train_input, f_train_output, verbose=True):

        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = self.engine.train
            iterator = self.engine.kf_train
        elif whichset == 'valid':
            tags = self.engine.valid
            iterator = self.engine.kf_valid
        elif whichset == 'test':
            tags = self.engine.test
            iterator = self.engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = np.sum([len(index) for index in iterator])
        self.engine.random_valid_idx()
        while(1):
            x, mask, ctx, ctx_mask = data_engine.prepare_data(
                self.engine, whichset, flag_rand='_valid')
            if x == None : 
                break
            pred_probs = sess.run(-f_train_output['cost'], 
                                    feed_dict={f_train_input['sen']:x, 
                                               f_train_input['sen_mask']: mask,
                                               f_train_input['feat']:ctx,
                                               f_train_input['feat_mask']:ctx_mask})
            pred_probs = np.array([pred_probs])
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(x)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples'%(
                             n_done, n_samples))
                sys.stdout.flush()
        print
        probs = flatten_list_of_list(probs)
        NLL = flatten_list_of_list(NLL)
        L = flatten_list_of_list(L)
        perp = 2**(np.sum(NLL) / np.sum(L) / np.log(2))
        return -1 * np.mean(probs), perp
    
    def print_trainable_variable(self,sess):
        variables_names =[v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            print(k, v)
        return True
        
    def train(self,
              random_seed=1234,
              reload_=False,
              verbose=True,
              debug=True,
              save_model_dir='',
              from_dir=None,
              # dataset
              dataset='youtube2text',
              video_feature='googlenet',
              num_frames=10,
              OutOf=240,
              # network
              dim_word=256, # word vector dimensionality
              dim_lstm_sen=-1, # context vector dimensionality, auto set
              dim_lstm_feat_sen=512,
              dim_feat=1024,
              n_layers_out=1,
              n_layers_init=1,
              encoder='none',
              encoder_dim=100,
              prev2out=False,
              ctx2out=False,
              selector=False,
              num_words=100000,
              max_len_sen=100, # maximum length of the description
              use_dropout=False,
              drop_out_rate=0.5,
              isGlobal=False,
              # training
              init_scale=0.04,
              patience=10,
              max_epochs=5000,
              decay_c=0.,
              alpha_c=0.,
              alpha_entropy_r=0.,
              lrate=0.01,
              optimizer='adadelta',
              clip_c=2.,
              # minibatch
              batch_size = 64,
              valid_batch_size = 64,
              dispFreq=100,
              validFreq=10,
              saveFreq=10, # save the parameters after every saveFreq updates
              sampleFreq=10, # generate some samples after every sampleFreq updates
              # metric
              metric='blue'
              ):
        self.rng_numpy, self.rng_theano = get_two_rngs()

        model_options = locals().copy()
        if 'self' in model_options:
            del model_options['self']
        model_options = validate_options(model_options)
        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
            pkl.dump(model_options, f)

        print 'Loading data'
        self.engine = data_engine.Movie2Caption('attention', dataset,
                                           video_feature,
                                           batch_size, valid_batch_size,
                                           max_len_sen, num_words,
                                           num_frames, OutOf,random=self.rng_numpy)
        model_options['dim_feat'] = self.engine.ctx_dim

        print 'init params'
        t0 = time.time()

        # reloading

        self.tf_caption_model = Video_Caption_MP_RNN(model_options)
        
        print 'buliding model'
        f_train_input, f_train_output, model_weight = self.tf_caption_model.build_model()

        print 'buliding sampler'
        f_init_inputs, f_init_outputs, \
        f_next_inputs, f_next_outputs = self.tf_caption_model.build_sample_model()
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # before any regularizer
        with tf.device("/cpu:0"):
            saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state(from_dir)
        if reload_:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, "", True)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        
        temp_var = set(tf.global_variables())
        print 'temp_var: ',tf.global_variables()
        
        optimizer = tf.train.AdadeltaOptimizer(model_options['lrate'])
        gvs = optimizer.compute_gradients(f_train_output['loss'])
        print '-------------gvs------------- \n',gvs
        clip_gvs = [(tf.clip_by_norm(grad, model_options['clip_c']), var) for grad, var in gvs if grad is not None]
        print '-------------clip_gvs------------- \n',clip_gvs
        train_op = optimizer.apply_gradients(clip_gvs)

        print 'compilation took %.4f sec'%(time.time()-t0)
        print 'Optimization'
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp_var))
        history_errs = []
        # reload history

        #self.print_trainable_variable(sess)
        
        bad_counter = 0
        processes = None
        queue = None
        rqueue = None
        shared_params = None

        uidx = 0
        best_v_sess = None
        uidx_best_blue = 0
        uidx_best_valid_err = 0
        estop = False
        best_blue_valid = 0
        best_valid_err = 999
        best_valid_b4 = 0.0
        for eidx in xrange(max_epochs):
            n_samples = 0
            train_costs = []
            grads_record = []
            print 'Epoch ', eidx
            self.engine.random_idx()
            while(1):
                uidx += 1 
                pd_start = time.time()
                x, mask, ctx, ctx_mask = data_engine.prepare_data(
                    self.engine, tset='train')
                
                if ctx is None :
                    print('ctx is None, epoch finished!\n')
                    break
                '''
                if ctx.shape[0] != model_options['batch_size']:
                    print 'Minibatch with zero sample under length ', max_len_sen
                    print 'ctx.shape[0] is :', ctx.shape[0]
                    break
                '''
                n_samples += len(x)
                pd_duration = time.time() - pd_start
                
                ud_start = time.time()
                _,loss,cost,probs = sess.run([train_op,f_train_output['loss'],f_train_output['cost'],
                                               f_train_output['pro_word']],
                                    feed_dict={f_train_input['sen']:x, 
                                               f_train_input['sen_mask']: mask,
                                               f_train_input['feat']:ctx,
                                               f_train_input['feat_mask']:ctx_mask})
                '''
                grads = rvals[2:]
                grads, NaN_keys = grad_nan_report(grads, tparams)
                if len(grads_record) >= 5:
                    del grads_record[0]
                grads_record.append(grads)
                if NaN_keys != []:
                    print 'grads contain NaN'
                    import pdb; pdb.set_trace()
                '''
                if np.isnan(loss) or np.isinf(loss):
                    print 'NaN detected in loss'
                    import pdb; pdb.set_trace()
                # update params
                ud_duration = time.time() - ud_start

                train_costs.append(loss)

                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Train loss mean so far',loss, \
                       'Train cost mean so far', cost,\
                       'fetching data time spent (sec)', np.round(pd_duration,3), \
                       'update time spent (sec)', np.round(ud_duration,3)
                       #'\n probs:', np.array(probs).transpose([1,0]),\
                       #'\n x:',pdf_r 
                    #np.save('probs.npy',probs)
                    #np.save('mask.npy',mask)
                if np.mod(uidx, saveFreq) == 0:
                    pass

                if np.mod(uidx, sampleFreq) == 0:
                    print '------------- sampling from train ----------'
                    self.predict.sample_execute(self.engine, model_options, sess, self.tf_caption_model,
                                                f_init_inputs,f_init_outputs,
                                                f_next_inputs, f_next_outputs, x, ctx, ctx_mask)

                    print '------------- sampling from valid ----------'
                    self.engine.random_valid_idx()
                    x_s, mask_s, ctx_s, mask_ctx_s = data_engine.prepare_data(self.engine, 'valid',flag_rand='_valid')
                    self.predict.sample_execute(self.engine, model_options, sess, self.tf_caption_model,
                                                f_init_inputs,f_init_outputs,
                                                f_next_inputs, f_next_outputs, x_s, ctx_s, mask_ctx_s)
                    # end of sample

                if validFreq != -1 and np.mod(uidx, validFreq) == 0:
                    t0_valid = time.time()

                    #current_params = unzip(tparams)
                    with tf.device("/cpu:0"):
                        saver.save(sess, os.path.join(save_model_dir, 'cur_model'))

                    train_err = -1
                    train_perp = -1
                    valid_err = -1
                    valid_perp = -1
                    test_err = -1
                    test_perp = -1
                    if not debug:
                        # first compute train cost
                        if 0:
                            print 'computing cost on trainset'
                            train_err, train_perp = self.pred_probs(
                                    'train', model_options, sess,  f_train_input, f_train_output,
                                    verbose=model_options['verbose'])
                        else:
                            train_err = 0.
                            train_perp = 0.
                        if 1:
                            print 'validating...'
                            valid_err, valid_perp = self.pred_probs(
                                'valid', model_options,  sess,  f_train_input, f_train_output,
                                verbose=model_options['verbose'],
                                )
                        else:
                            valid_err = 0.
                            valid_perp = 0.
                        if 1:
                            print 'testing...'
                            test_err, test_perp = self.pred_probs(
                                'test', model_options, sess, f_train_input, f_train_output,
                                verbose=model_options['verbose']
                                )
                        else:
                            test_err = 0.
                            test_perp = 0.

                    mean_ranking = 0
                    blue_t0 = time.time()
                    scores, processes, queue, rqueue, shared_params = \
                        metrics.compute_score(model_type='attention',
                                              options=model_options,
                                              engine=self.engine,
                                              sess = sess, model = self.tf_caption_model,
                                              save_dir=save_model_dir,
                                              beam=5, n_process=5,
                                              whichset='both',
                                              on_cpu=False,
                                              processes=processes, queue=queue, rqueue=rqueue,
                                              shared_params=shared_params, metric=metric,
                                              one_time=False,
                                              f_init_inputs=f_init_inputs,f_init_outputs=f_init_outputs, 
                                              f_next_inputs=f_next_inputs,f_next_outputs=f_next_outputs
                                              )

                    valid_B1 = scores['valid']['Bleu_1']
                    valid_B2 = scores['valid']['Bleu_2']
                    valid_B3 = scores['valid']['Bleu_3']
                    valid_B4 = scores['valid']['Bleu_4']
                    valid_Rouge = scores['valid']['ROUGE_L']
                    valid_Cider = scores['valid']['CIDEr']
                    valid_meteor = scores['valid']['METEOR']
                    test_B1 = scores['test']['Bleu_1']
                    test_B2 = scores['test']['Bleu_2']
                    test_B3 = scores['test']['Bleu_3']
                    test_B4 = scores['test']['Bleu_4']
                    test_Rouge = scores['test']['ROUGE_L']
                    test_Cider = scores['test']['CIDEr']
                    test_meteor = scores['test']['METEOR']
                    print 'computing meteor/blue score used %.4f sec, '\
                          'blue score: %.1f, meteor score: %.1f'%(
                    time.time()-blue_t0, valid_B4, valid_meteor)
                    history_errs.append([eidx, uidx, train_err, train_perp,
                                         valid_perp, test_perp,
                                         valid_err, test_err,
                                         valid_B1, valid_B2, valid_B3,
                                         valid_B4, valid_meteor, valid_Rouge, valid_Cider,
                                         test_B1, test_B2, test_B3,
                                         test_B4, test_meteor, test_Rouge, test_Cider])
                    np.savetxt(save_model_dir+'train_valid_test.txt',
                                  history_errs, fmt='%.3f')
                    print 'save validation results to %s'%save_model_dir
                    # save best model according to the best blue or meteor
                    
                    if len(history_errs) > 1 and valid_B4 >= best_valid_b4 and valid_meteor >= 0.33:
                        best_valid_b4 = valid_B4
                        print 'Saving to %s...'%save_model_dir,
                        with tf.device("/cpu:0"):
                            saver.save(sess, os.path.join(save_model_dir, 'best_v_b4_model'))
                            
                    if len(history_errs) > 1 and test_B4 >= np.array(history_errs)[:-1,18].max():
                        print 'Saving to %s...'%save_model_dir,
                        with tf.device("/cpu:0"):
                            saver.save(sess, os.path.join(save_model_dir, 'best_t_b4_model'))
                                       
                    if len(history_errs) > 1 and test_meteor >= np.array(history_errs)[:-1,19].max():
                        print 'Saving to %s...'%save_model_dir,
                        with tf.device("/cpu:0"):
                            saver.save(sess, os.path.join(save_model_dir, 'best_t_m_model'))
                                       
                    if len(history_errs) > 1 and valid_err <= np.array(history_errs)[:-1,6].min():
                        bad_counter = 0
                        best_v_sess = sess
                        best_valid_err = valid_err
                        uidx_best_valid_err = uidx
                        with tf.device("/cpu:0"):
                            saver.save(sess, os.path.join(save_model_dir, 'best_v_er_model'))
                                       
                        print 'Saving to %s...'%save_model_dir,
                        np.savez(
                            save_model_dir+'history_errs.npz',
                            history_errs=history_errs)
                        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
                            pkl.dump(model_options, f)
                        print 'Done'
                    elif len(history_errs) > 1 and valid_err >= np.array(history_errs)[:-1,6].min():
                        bad_counter += 1
                        print 'history best ',np.array(history_errs)[:,6].min()
                        print 'bad_counter ',bad_counter
                        print 'patience ',patience
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if self.channel:
                        self.channel.save()

                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, \
                          'best valid err so far',best_valid_err
                    print 'valid took %.2f sec'%(time.time() - t0_valid)
                    # end of validatioin
                if debug:
                    break
            if estop:
                break
            if debug:
                break

            # end for loop over minibatches
            print 'This epoch has seen %d samples, train cost %.2f'%(
                n_samples, np.mean(train_costs))
        # end for loop over epochs
        print 'Optimization ended.'

        print 'stopped at epoch %d, minibatch %d, '\
              'curent Train %.2f, current Valid %.2f, current Test %.2f '%(
               eidx, uidx, np.mean(train_err), np.mean(valid_err), np.mean(test_err))
        with tf.device("/cpu:0"):
            saver.save(best_v_sess, os.path.join(save_model_dir, 'best_model'))
        np.savez(save_model_dir+'model_error.npz',
                 train_err=train_err,
                 valid_err=valid_err, test_err=test_err, history_errs=history_errs)

        if history_errs != []:
            history = np.asarray(history_errs)
            best_valid_idx = history[:,6].argmin()
            np.savetxt(save_model_dir+'train_valid_test.txt', history, fmt='%.4f')
            print 'final best exp ', history[best_valid_idx]

        return train_err, valid_err, test_err


def train_from_scratch(state, channel):
    t0 = time.time()
    print 'training an attention model'
    model = Attention(channel)
    model.train(**state.attention)
    print 'training time in total %.4f sec'%(time.time()-t0)

