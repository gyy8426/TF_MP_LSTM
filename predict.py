from utils import *
import copy


class Predict(object):

            
    def sample_execute(self, engine, options, sess, model,
                       f_init_inputs,f_init_outputs, f_next_inputs, f_next_outputs, 
                       x, ctx, mask_ctx, trng=None):
        stochastic = False
        for jj in xrange(np.minimum(10, x.shape[0])):
            sample, score, _, _ = model.gen_sen_beam_search(sess, ctx[jj], mask_ctx[jj],
                                                            f_init_inputs,f_init_outputs,
                                                            f_next_inputs,f_next_outputs,
                                                            k=5, maxlen=30)
            if not stochastic:
                best_one = np.argmin(score)
                sample = sample[best_one]
            else:
                sample = sample
            print 'Truth ', jj, ': ',
            for vv in x[jj,:]:
                if vv == 0:
                    break
                if vv in engine.ix_word:
                    print engine.ix_word[vv],
                else:
                    print 'UNK',
            print
            for kk, ss in enumerate([sample]):
                print 'Sample (', jj, ') ', ': ',
                for vv in ss:
                    if vv == 0:
                        break
                    if vv in engine.ix_word:
                        print engine.ix_word[vv],
                    else:
                        print 'UNK',
            print
