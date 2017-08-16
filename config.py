from jobman import DD

RAB_DATASET_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/predatas/'
RAB_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/Resnet/'
OPT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/OptFlow_Resnet/'
ACT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/TS_LSTM/'
RAB_EXP_PATH = '/home/guoyuyu/results/youtube/tf_MP_Double_LSTM_Res/'

config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': True,
    'attention': DD({
        'reload_': False,
        'verbose': True,
        'debug': False,
        'save_model_dir': RAB_EXP_PATH + 'save_dir/',
        'from_dir': RAB_EXP_PATH + 'from_dir/',
        # dataset
        'dataset': 'youtube2text', 
        'video_feature': 'googlenet', #opt_res #opt_res_act #googlenet #res_act
        'num_frames':28, # 26 when compare
        'OutOf':None,
        # network
        'dim_word':512,#468, # 474
        'dim_lstm_sen': 512,
        'dim_lstm_feat_sen': 512,
        'dim_feat':-1,# auto set
        'n_layers_out':1, # for predicting next word
        'n_layers_init':0,
        'encoder_dim': 1024,#300,
        'prev2out':True,
        'ctx2out':True,
        'selector':True,
        'num_words':20000,
        'max_len_sen':35, # max length of the descprition
        'use_dropout':True,
        'drop_out_rate': 0.5,
        'isGlobal': True,
        # training
        'init_scale':0.04,
        'patience':20,
        'max_epochs':500,
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.1,#0.1
        'optimizer':'adadelta',
        'clip_c': 100.,
        # minibatches
        'batch_size': 64, # for trees use 25
        'valid_batch_size':64,
        'dispFreq':10,
        'validFreq':2000, 
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        }),
    })
