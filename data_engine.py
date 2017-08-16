import time
import config
import utils
import os
import numpy as np


class Movie2Caption(object):
            
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,
                 n_frames=None, outof=None, random=None
                 ):
        self.signature = signature
        self.model_type = model_type
        self.video_feature = video_feature
        self.maxlen = maxlen
        self.n_words = n_words
        self.K = n_frames
        self.OutOf = outof
        self.mb_size_train = mb_size_train
        self.mb_size_valid = mb_size_test
        self.mb_size_test = mb_size_test
        self.non_pickable = []
        self.random = random
        self.load_data()
        
    def _filter_googlenet(self, vidID):
        feat_file = os.path.join(self.FEAT_ROOT, vidID + '.npy')
        feat = np.load(feat_file)
        feat = self.get_sub_frames(feat)
        return feat
        
    def _filter_opt(self, vidID):
        feat_file = os.path.join(self.OPT_FEAT_ROOT, vidID + '.npy')
        feat = np.load(feat_file)
        if self.signature == 'msr-vtt':
            feat = feat[:-1,:]
        feat = self.get_sub_frames(feat)
        return feat
        
    def _filter_act(self, vidID):
        feat_file = os.path.join(self.ACT_FEAT_ROOT, vidID + '.npy')
        feat = np.load(feat_file)
        return feat

    def get_video_features(self, vidID):
        if self.video_feature == 'googlenet':
            y = self._filter_googlenet(vidID)

        elif self.video_feature == 'opt_res':
            y_1 = self._filter_googlenet(vidID)
            y_2 = self._filter_opt(vidID)
            y = np.concatenate([y_1,y_2],axis=-1)
        elif self.video_feature == 'opt_res_act':
            y_1 = self._filter_googlenet(vidID)
            y_2 = self._filter_opt(vidID)
            y_3 = self._filter_act(vidID)
            # y_3 shape: 1,512
            y_3 = y_3.repeat(y_1.shape[0],axis=0)
            y = np.concatenate([y_1,y_2,y_3],axis=-1)
        elif self.video_feature == 'res_act':
            y_1 = self._filter_googlenet(vidID)
            y_3 = self._filter_act(vidID)
            # y_3 shape: 1,512
            y_3 = y_3.repeat(y_1.shape[0],axis=0)
            y = np.concatenate([y_1,y_3],axis=-1)
        else:
            raise NotImplementedError()
        return y  

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = np.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = np.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = np.array_split(range(n_frames), self.K)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = np.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = np.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = np.concatenate([frames, eos], axis=0)
        return frames
    
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take K of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if self.OutOf:
            raise NotImplementedError('OutOf has to be None')
            frames_ = frames[:self.OutOf]
            if len(frames_) < self.OutOf:
                frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
        else:
            if len(frames) < self.K:
                #frames_ = self.add_end_of_video_frame(frames)
                frames_ = self.pad_frames(frames, self.K, jpegs)
            else:

                frames_ = self.extract_frames_equally_spaced(frames, self.K)
                #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_

    def prepare_data_for_blue(self, whichset):
        # assume one-to-one mapping between ids and features
        feats = []
        feats_mask = []
        if whichset == 'valid':
            ids = self.valid_ids
        elif whichset == 'test':
            ids = self.test_ids
        elif whichset == 'train':
            ids = self.train_ids
        for i, vidID in enumerate(ids):
            feat = self.get_video_features(vidID)
            feats.append(feat)
            feat_mask = self.get_ctx_mask(feat)
            feats_mask.append(feat_mask)
        return feats, feats_mask
    
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
        
    def random_kf_idx(self):
        self.kf_train = utils.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = utils.generate_minibatch_idx(
            len(self.valid), self.mb_size_test)
        self.kf_test = utils.generate_minibatch_idx(
            len(self.test), self.mb_size_test)
            
    def flag_sub_ids(self, ids):
        flag_sub_ids = {}
        for id in ids:
            flag_sub_ids[id] = 1
        return flag_sub_ids
        
    def random_idx(self):
        self.train_ids_random = list(self.train)
        self.random.shuffle(self.train_ids_random)
        self.valid_ids_random = list(self.valid)
        self.random.shuffle(self.valid_ids_random)
        self.test_ids_random = list(self.test)
        self.random.shuffle(self.test_ids_random)
        return True
        
    def random_valid_idx(self):
        self.train_ids_random_valid = list(self.train)
        self.random.shuffle(self.train_ids_random)
        self.valid_ids_random_valid = list(self.valid)
        self.random.shuffle(self.valid_ids_random)
        self.test_ids_random_valid = list(self.test)
        self.random.shuffle(self.test_ids_random)
        return True
        
    def load_data(self):

        print 'loading youtube2text %s features'%self.video_feature
        dataset_path = config.RAB_DATASET_BASE_PATH
        self.train = utils.load_pkl(dataset_path + 'train.pkl')
        self.valid = utils.load_pkl(dataset_path + 'valid.pkl')
        self.test = utils.load_pkl(dataset_path + 'test.pkl')
        self.CAP = utils.load_pkl(dataset_path + 'CAP.pkl')
        self.FEAT_ROOT = config.RAB_FEATURE_BASE_PATH
        self.OPT_FEAT_ROOT = config.OPT_FEATURE_BASE_PATH
        self.ACT_FEAT_ROOT = config.ACT_FEATURE_BASE_PATH
        if self.signature == 'youtube2text':
            self.train_ids = ['vid%s'%i for i in range(1,1201)]
            self.valid_ids = ['vid%s'%i for i in range(1201,1301)]
            self.test_ids = ['vid%s'%i for i in range(1301,1971)]
        elif self.signature == 'msr-vtt':
            self.train_ids = ['video%s'%i for i in range(0,6513)]
            self.valid_ids = ['video%s'%i for i in range(6513,7910)]
            self.test_ids = ['video%s'%i for i in range(7910,10000)]
        else:
            raise NotImplementedError()
        self.random_idx()
        self.word_ix = utils.load_pkl(dataset_path + 'worddict.pkl')
        self.ix_word = dict()
        # word_ix start with index 2
        for kk, vv in self.word_ix.iteritems():
            self.ix_word[vv] = kk
        self.ix_word[0] = '<eos>'
        self.ix_word[1] = 'UNK'
        
        if self.video_feature == 'googlenet':
            self.ctx_dim = 2048
        elif self.video_feature == 'opt_res':
            self.ctx_dim = 2048 * 2
        elif self.video_feature == 'opt_res_act':
            self.ctx_dim = 2048 * 2 + 512
        elif self.video_feature == 'res_act':
            self.ctx_dim = 2048 + 512
        else:
            raise NotImplementedError()
        self.kf_train = utils.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = utils.generate_minibatch_idx(
            len(self.valid), self.mb_size_valid)
        self.kf_test = utils.generate_minibatch_idx(
            len(self.test), self.mb_size_test)


def prepare_data(engine, tset='train',flag_rand=''):

    seqs = []
    feat_list = []
    '''
    if len(eval('engine.'+tset+'_ids_random'+flag_rand)) < eval('engine.'+'mb_size_'+tset):
        return None,None,None,None
    '''
    def get_words(vidID, capID):
        if engine.signature == 'youtube2text':
            caps = engine.CAP[vidID]
            rval = None
            for cap in caps:
                if cap['cap_id'] == capID:
                    rval = cap['tokenized'].split(' ')
                    break
        elif engine.signature == 'msr-vtt':
            rval = engine.CAP[int(capID)]['caption'].split(' ')
            rval = [w for w in rval if w != '']
        assert rval is not None
        return rval
    count = 0
    for ID in eval('engine.'+tset+'_ids_random'+flag_rand):
        # load GNet feature
        eval('engine.'+tset+'_ids_random'+flag_rand).remove(ID)
        vidID, capID = ID.split('_')
        words = get_words(vidID, capID)
        if len(words) < engine.maxlen:
            seqs.append([engine.word_ix[w]
                         if w in engine.word_ix else 1 for w in words])
            feat = engine.get_video_features(vidID)
            feat_list.append(feat)
            count = count + 1
        if count == eval('engine.'+'mb_size_'+tset):
            break
            
    if  feat_list == []:
        return None,None,None,None
        
    lengths = [len(s) for s in seqs]
    
    y = np.asarray(feat_list)
    y_mask = engine.get_ctx_mask(y)
    n_samples = len(seqs)
    maxlen = engine.maxlen
    x = np.zeros((n_samples, maxlen)).astype('int64')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]+1] = 1.
    return x, x_mask, y, y_mask


def test_data_engine():
    from sklearn.cross_validation import KFold
    video_feature = 'googlenet' 
    out_of = None
    maxlen = 100
    mb_size_train = 64
    mb_size_test = 128
    maxlen = 50
    n_words = 30000 # 25770 
    signature = 'youtube2text' #'youtube2text'
    engine = Movie2Caption('attention', signature, video_feature,
                           mb_size_train, mb_size_test, maxlen,
                           n_words,
                           n_frames=26,
                           outof=out_of, random = np.random.RandomState(1234))
    '''
    (self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,
                 n_frames=None, outof=None
                 ):
    '''
    i = 0
    t = time.time()
    for i in engine.train_ids_random:
        print i
        
    while(1):
        x, mask, ctx, ctx_mask = prepare_data(engine, tset='train')
        if x == None:
            print 'Epoch finished!'
            engine.random_idx()
            print 'len: ',len(engine.train_ids_random)
            tent = input("input:")
            continue
        print 'x shape:', x.shape
    '''
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids)
        print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
        if i == 10:
            break
    '''
    print 'used time %.2f'%(time.time()-t)
if __name__ == '__main__':
    test_data_engine()


