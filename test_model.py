from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
import time
from layers import Layers
import data_engine
import metrics
from optimizers import *
import numpy as np
from predict import *


def validate_options(options):
    if options['ctx2out']:
        warnings.warn('Feeding context to output directly seems to hurt.')
    if options['dim_word'] > options['mu_dim']:
        warnings.warn('dim_word should only be as large as mu_dim.')
    return options


class Attention():
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

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        ctx_dim = options['ctx_dim']

        # decoder: LSTM
        params = self.layers.get_layer('lstm')[0](params, nin=options['dim_word'],
                                                  dim=options['tu_dim'], prefix='tu_lstm')
        params = self.layers.get_layer('lstm_cond')[0](options, params, nin=options['tu_dim'],
                                                       dim=options['mu_dim'], dimctx=ctx_dim,
                                                       prefix='mu_lstm')

        # readout
        params = self.layers.get_layer('ff')[0](params, nin=options['mu_dim'], nout=options['n_words'],
                                                prefix='ff_logit_lstm')

        return params

    def build_model(self, tparams, options):
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = tparams['Wemb'][x.flatten()].reshape(
                [n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        ctx_ = ctx
        counts = mask_ctx.sum(-1).dimshuffle(0,'x')
        ctx_mean = ctx_.sum(1)/counts

        # decoder
        tu_lstm = self.layers.get_layer('lstm')[1](tparams, emb, mask=mask, prefix='tu_lstm')
        mu_lstm = self.layers.get_layer('lstm_cond')[1](options, tparams, tu_lstm[0],
                                                        mask=mask, context=ctx_mean,
                                                        one_step=False,
                                                        trng=trng,
                                                        use_noise=use_noise,
                                                        prefix='mu_lstm')

        proj_h = mu_lstm[0]

        if options['use_dropout']:
            proj_h = self.layers.dropout_layer(proj_h, use_noise, trng)

        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](tparams, proj_h, activ='linear',
                                               prefix='ff_logit_lstm')
        logit_shp = logit.shape
        # (t*m, n_words)
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))

        # cost
        x_flat = x.flatten() # (t*m,)
        cost = -tensor.log(probs[T.arange(x_flat.shape[0]), x_flat] + 1e-8)
        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * mask).sum(0)

        extra = [probs]

        return trng, use_noise, x, mask, ctx, mask_ctx, cost, extra

    def pred_probs(self, whichset, f_log_probs, verbose=True):

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
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = data_engine.prepare_data(
                self.engine, tag)
            pred_probs = f_log_probs(x, mask, ctx, ctx_mask)
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
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
              K=10,
              OutOf=None,
              # network
              dim_word=256, # word vector dimensionality
              ctx_dim=-1, # context vector dimensionality, auto set
              tu_dim=512,
              mu_dim=1024,
              n_layers_out=1,
              n_layers_init=1,
              encoder='none',
              encoder_dim=100,
              prev2out=False,
              ctx2out=False,
              selector=False,
              n_words=100000,
              maxlen=100, # maximum length of the description
              use_dropout=False,
              isGlobal=False,
              # training
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
        from_dir = '/home/guoyuyu/results/youtube/MP_Double_LSTM_OptRes_1/save_dir/'
        model_options = np.load(from_dir+'model_options.pkl')
        if 'self' in model_options:
            del model_options['self']
        print 'Loading data'
        self.engine = data_engine.Movie2Caption('attention', model_options['dataset'],
                                            model_options['video_feature'],
                                           model_options['batch_size'], model_options['valid_batch_size'],
                                           model_options['maxlen'], model_options['n_words'],
                                           model_options['K'], model_options['OutOf'])
        model_options['ctx_dim'] = self.engine.ctx_dim

        print 'init params'
        t0 = time.time()
        params = self.init_params(model_options)
        
        model_saved = from_dir+'/model_best_test_blue4.npz' #model_best_test_blue4.npz
        assert os.path.isfile(model_saved)
        print "Reloading model params..."
        params = load_params(model_saved, params)

        tparams = init_tparams(params)
        if verbose:
            print tparams.keys

        trng, use_noise, x, mask, ctx, mask_ctx, cost, extra = \
            self.build_model(tparams, model_options)

        print 'buliding sampler'
        use_noise.set_value(0.)
        f_init, f_next = self.predict.build_sampler(self.layers, tparams, model_options, use_noise, trng)
        # before any regularizer
        '''
        print 'building f_log_probs'
        f_log_probs = theano.function([x, mask, ctx, mask_ctx], -cost,
                                      profile=False, on_unused_input='ignore')

        cost = cost.mean()
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        print 'compute grad'
        grads = tensor.grad(cost, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        lr = tensor.scalar(name='lr')
        print 'build train fns'
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads,
                                                  [x, mask, ctx, mask_ctx], cost,
                                                  extra + grads)

        print 'compilation took %.4f sec'%(time.time()-t0)
        print 'Optimization'

        history_errs = []
        # reload history
        if reload_:
            print 'loading history error...'
            history_errs = np.load(
                from_dir+'model_best_so_far.npz')['history_errs'].tolist()
        '''
        bad_counter = 0

        processes = None
        queue = None
        rqueue = None
        shared_params = None

        uidx = 0
        uidx_best_blue = 0
        uidx_best_valid_err = 0
        estop = False
        best_p = unzip(tparams)
        best_blue_valid = 0
        best_valid_err = 999
        current_params = unzip(tparams)
        blue_t0 = time.time()

        scores, processes, queue, rqueue, shared_params = \
            metrics.compute_score(model_type='attention',
                                  model_archive=current_params,
                                  options=model_options,
                                  engine=self.engine,
                                  save_dir=from_dir+'test_finish_model_',
                                  beam=5, n_process=5,
                                  whichset='test',
                                  on_cpu=False,
                                  processes=processes, queue=queue, rqueue=rqueue,
                                  shared_params=shared_params, metric=metric,
                                  one_time=False,
                                  f_init=f_init, f_next=f_next, model=self.predict
                                  )
        '''
        valid_B1 = scores['valid']['Bleu_1']
        valid_B2 = scores['valid']['Bleu_2']
        valid_B3 = scores['valid']['Bleu_3']
        valid_B4 = scores['valid']['Bleu_4']
        valid_Rouge = scores['valid']['ROUGE_L']
        valid_Cider = scores['valid']['CIDEr']
        valid_meteor = scores['valid']['METEOR']
        '''
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



if __name__ == '__main__':
    t0 = time.time()
    print 'training an attention model'
    model = Attention()
    model.train()
    print 'training time in total %.4f sec'%(time.time()-t0)

