#!/usr/bin/env python3
import numpy as np
import os
import json

np.random.seed(1993)
# ./main.py --lr 0.3 --frame-length 34 --frame-shift 9 --num-mel-bins 32 ../datasets/google_speech_commands/data/ traindir
WAV2VEC_MODEL_PATH = '/home/kolesov93/study/wav2vec_models/first.pt'

LANGUAGES = {
    'ru': {
        'words': 'один,два,три,четыре,пять,да,нет,спасибо,стоп,включи',
        'data': '/home/kolesov93/study/datasets/rus_data'
    },
    'en': {
        'words': 'yes,no,up,down,left,right,on,off,stop,go',
        'data': '/home/kolesov93/study/datasets/data'
    },
    'lt': {
        'words': 'ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki',
        'data': '/home/kolesov93/study/datasets/lt_data'
    }
}

MDLS = """
uns_repro_traindirs/226_en_fbank_res26_narrow_batch32_lr0.1515304162147572_lr_drop5.726836830289077_dev_every_batches2048/model_30720.mdl
uns_repro_traindirs/013_ru_fbank_res8_batch32_lr0.657779019924163_lr_drop1.1740103640154846_dev_every_batches2048/model_38912.mdl
uns_repro_traindirs/091_en_fbank_res15_limit20_batch16_lr0.023720595572496672_lr_drop3.305009741158604_dev_every_batches2048/model_16384.mdl
uns_repro_traindirs/128_en_fbank_res15_batch64_lr0.0035218067875324136_lr_drop1.4771781497955914_dev_every_batches512/model_8192.mdl
uns_repro_traindirs/309_ru_fbank_res15_limit20_batch16_lr0.00118615732695068_lr_drop1.3961030134620853_dev_every_batches2048/model_18432.mdl
""".strip().split()

LIMITS = [3, 5, 7, 10, 20]
LRS = [0.1, 0.01]
BATCH_SIZES = [16, 32, 64, 128]

MODELS = ['res15']

args = []
T = 200

for _ in range(T):
    limit = np.random.choice(LIMITS)

    lr = 10 ** np.random.uniform(-3., 0.)
    lr_drop = np.random.uniform(1.1, 10.0)
    dev_every_batches = 2 ** np.random.randint(6, 12)
    batch_size = 2 ** np.random.randint(4, 7)

    lang = np.random.choice(list(LANGUAGES.keys()))
    lang = 'lt'

    use_fbank = np.random.choice([False, True])
    use_fbank = True

    init_model = np.random.choice(MDLS)
    path_tokens = init_model.split('/')
    with open('/'.join(path_tokens[:-1] + ['options.json'])) as fin:
        model = json.load(fin)['model']
    new_args = {
        'lr': lr,
        'batch-size': batch_size,
        'wanted-words': LANGUAGES[lang]['words'],
        'language': lang,
        'lr-drop': lr_drop,
        'model': model,
        'dev-every-batches': dev_every_batches,
        'max-batches': '10000'
    }
    if np.random.random() < 0.5:
        new_args['initialize-body'] = init_model

    if limit is not None:
        new_args['limit'] = limit
    if use_fbank:
        new_args['use-fbank'] = None
    args.append(new_args)

def _make_traindir(args):
    tokens = []
    tokens.append(args['language'])
    if 'use-fbank' in args:
        tokens.append('fbank')
    else:
        tokens.append('wav2vec')
    tokens.append(args['model'])
    if 'limit' in args:
        tokens.append('limit{}'.format(args['limit']))
    if 'initialize-body' in args:
        tokens.append('uptrain')
    tokens.append('batch{}'.format(args['batch-size']))
    tokens.append('lr{}'.format(args['lr']))
    tokens.append('lr_drop{}'.format(args['lr-drop']))
    tokens.append('dev_every_batches{}'.format(args['dev-every-batches']))
    return '_'.join(tokens)

cmds = []
for i, cargs in enumerate(args):
    cmd = ['PYTHONPATH=/home/kolesov93/study/fairseq', 'python3.6', './main.py']
    traindir = _make_traindir(cargs)
    language = cargs.pop('language')
    for k, v in cargs.items():
        cmd.append('--' + k)
        if v:
            cmd.append(str(v))

    cmd.append(LANGUAGES[language]['data'])
    cmd.append('uns_lt_uptrain_small_limit_traindirs/{:03d}_{}'.format(i, traindir))
    cmds.append(' '.join(cmd))

for cmd in cmds:
    print(cmd)
