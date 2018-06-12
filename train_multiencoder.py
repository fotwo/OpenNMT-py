#!/usr/bin/env python

from __future__ import division
from collections import defaultdict

import argparse
import glob
import os
import sys
from datetime import datetime
import random

import torch
from torch import cuda
import torch.nn as nn
import onmt
import onmt.ModelConstructorForMultiEncoder
from onmt.TrainerForMultiEncoder import TrainerForMultiEncoder

from onmt.io.TextDataset import TextDataset
from onmt.io.AudioDataset import AudioDataset
from onmt.Utils import use_gpu



def data_opts(parser):
    group=parser.add_argument_group('Data - Pretrained Embeddings')
    group.add_argument('-pre_word_vecs_enc',
                       help='''If a valid path is specified, then this will
                       load pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.''')
    group.add_argument('-pre_word_vecs_dec',
                       help='''If a valid path is specified, then this will
                       load pretrained embeddings on the decoder side.
                       See README for specific formatting instructions.''')
    group.add_argument('-fix_word_vecs_enc',
                       action='store_true',
                       help='''Fix word embeddings on the encoder side.''')
    group.add_argument('-fix_word_vecs_dec',
                       action='store_true',
                       help='''Fix word embeddings on the decoder side.''')


    group=parser.add_argument_group('Data - Speech')
    group.add_argument('-sample_rate',type=int,default=16000,
                       help='''Sample rate.''')
    group.add_argument('-window_size',type=float,default=0.02,
                       help='''Window size for spectrogram in seconds.''')


def train_opts(parser):
    group=parser.add_argument_group('General')
    group.add_argument('-data', required=True,
                       help='''Path to input data.''')
    group.add_argument('-save_model', default='model',
                       help='''Path to save model.''')
    group.add_argument('-gpuid',default=[],nargs='+',type=int,
                       help='''Use CUDA on the listed devices.''')
    group.add_argument('-seed',type=int,default=-1,
                       help='''Random seed used for the experiments reproducibility.''')

    # Initialization options.
    group=parser.add_argument_group('Initialization.')
    group.add_argument('-start_epoch',type=int,default=1,
                       help='''The epoch from which to start.''')
    group.add_argument('-param_init',type=float,default=0.1,
                       help='''Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization.''')
    group.add_argument('-param_init_glorot',action='store_true',
                       help='''Init parameters with xavier_uniform. Required for transformer.''')
    group.add_argument('-train_from',type=str, default='',
                       help='''If training from a checkpoint then this is the path to the
                       pretrained model's state_dict.''')


def model_opts(parser):
    '''
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    '''
    group=parser.add_argument_group('Model - Embeddings')
    group.add_argument('-src_word_vec_size',type=int,default=500,
                       help='''Word embedding size of src.''')
    group.add_argument('-tgt_word_vec_size',type=int,default=500,
                       help='''Word embedding size of tgt.''')
    group.add_argument('-word_vec_size',type=int,default=-1,
                       help='''Word embedding size for src and tgt.''')
    group.add_argument('-position_encoding',action='store_true',
                       help='''Use as sin to mark relative words position.
                       Necessary for non-RNN style models.''')
    group.add_argument('-share_embeddings',action='store_true',
                       help='''Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary
                       for this option. Not implemented.''')
    group.add_argument('-share_decoder_embeddings',action='store_true',
                       help='''Use a shared weight matrix for the input and
                       output word embeddings in the decoder.''')


    group=parser.add_argument_group('Model - Embedding Features.')
    group.add_argument('-feat_merge',type=str,default='concat',
                       choices=['concat','sum','mlp'],
                       help='''Merge action for incorporating features embedding.
                       Options [concat|sum|mlp].''')
    group.add_argument('-feat_vec_size',type=int,default=-1,
                       help='''If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.''')
    group.add_argument('-feat_vec_exponent',type=float,default=0.7,
                       help='''If -feat_merge_size is not set, feature embedding sizes
                       will be set to N^feat_vec_exponent
                       where N is the size of dictionary belonging to that feature.''')


    group=parser.add_argument_group('Model - MultiEncoder-Decoder')
    group.add_argument('-model_type',default='multiencoder',
                       help='''Type of encoder model to use.
                       Here we only only only support text-audio multiencoder.
                       So the value must be 'multiencoder'.''')
    group.add_argument('-encoder_type', type=str, default='rnn',
                       choices=['rnn', 'brnn'],
                       help='''Type of encoder layer to use.
                       Need more work to expend the options.[rnn|brnn].''')
    group.add_argument('-decoder_type',type=str,default='rnn',
                       choices=['rnn'],
                       help='''Type of decoder layer to use.
                       Need more work to expand the options.[rnn].''')
    group.add_argument('-layers',type=int,default=-1,
                       help='''Number of layers in enc/dec.''')
    group.add_argument('-enc_layers',type=int,default=2,
                       help='''Number of layers in encoder.''')
    group.add_argument('-dec_layers',type=int,default=2,
                       help='''Number of layers in decoder.''')
    group.add_argument('-rnn_size',type=int,default=500,
                       help='''Size of rnn hidden states.''')
    group.add_argument('-cnn_kernel_width',type=int,default=3,
                       help='''Size of windows in the cnn, the kernel_size
                       is (cnn_kernel_width, 1) in conv layer.''')

    group.add_argument('-input_feed',type=int,default=1,
                       help='''Feed the context vector at each time step
                       as additional input (via concatenating with the
                       word embeddings) to the decoder.''')
    group.add_argument('-bridge',action='store_true',
                       help='''Have an additional layer between the last
                       encoder state and the first decoder state.''')
    group.add_argument('-rnn_type',type=str,default='LSTM',
                       choices=['LSTM','GRU'],
                       help='''The gate type to use in the RNNs.''')
    group.add_argument('-brnn',action=DeprecateAction,
                       help='''Deprecated, use 'encoder_type'.''')

    group.add_argument('-context_gate',type=str,default=None,
                       help='''Not be used. Need more work.''')


    group=parser.add_argument_group('Model - Attention')
    group.add_argument('-global_attention',type=str,default='multi',
                       help='''The attention type to use for multiple encoders.
                       Only implement general method.''')

    group.add_argument('-copy_attn',action='store_true',
                       help='''Train copy attention layer. Not implemented.''')
    group.add_argument('-copy_attn_force',action='store_true',
                       help='''When avaiable, train to copy. Not implemented.''')
    group.add_argument('-reuse_copy_attn',action='store_true',
                       help='''Reuse standard attention for copy. Not implemented.''')
    group.add_argument('-copy_loss_by_seqlength',action='store_true',
                       help='''Divide copy loss by length of sequence.''')
    group.add_argument('-converage_attn',action='store_true',
                       help='''Train a converage attention layer. Not implemented.''')
    group.add_argument('-lambda_coverage',type=float,default=1,
                       help='''Lambda value for coverage.''')

def optim_opts(parser):
    group=parser.add_argument_group('Optimization')
    group.add_argument('-batch_size',type=int,default=64,
                       help='''Maximum batch size for training.''')
    group.add_argument('-batch_type',default='sents',
                       choices=['sents','tokens'],
                       help='''Batch grouping for batch_size. Standard
                       is sents. Tokens will do dynamic batching.''')
    group.add_argument('-normalization',default='sents',
                       choices=['sents','tokens'],
                       help='''Normalization method of the gradient.''')
    group.add_argument('-accum_count',type=int,default=1,
                       help='''Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommanded for Transformer.''')
    group.add_argument('-valid_batch_size',type=int,default=32,
                       help='Maximum batch size for validation.')
    group.add_argument('-max_generator_batches',type=int,default=32,
                       help='''Maximum batches of words in a sequence
                        to run the generator on in parallel. Higher
                        is faster, but uses more memory.''')
    group.add_argument('-epochs',type=int,default=13,
                       help='''Number of training epochs.''')
    group.add_argument('-optim',default='sgd',
                       choices=['sgd','adagrad','adadelta', 'adam',
                                'sparseadam'],
                       help='''Optimization method.''')
    group.add_argument('-adagrad_accumulator_init',type=float,default=0,
                       help='''Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).''')
    group.add_argument('-max_grad_norm',type=float,default=5,
                       help='''If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm.''')
    group.add_argument('-dropout',type=float,default=0.3,
                       help='''Dropout probability; applied in LSTM stacks.''')
    group.add_argument('-truncated_decoder',type=int,default=0,
                       help='''Truncated bptt.''')
    group.add_argument('-adam_beta1',type=float,default=0.9,
                       help='''The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourge changing this value from the
                       default without due consideration.''')
    group.add_argument('-adam_beta2',type=float,default=0.999,
                       help='''The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as
                       this is the value suggested by the original paper
                       describing Adam, and is also the value adopted in other
                       frameworks, such as tensorflow and keras.
                       Whereas recently the paper 'Attention is All You Need'
                        suggested a value of 0.98 for beta2, this parameter may
                        not work well for normal models / default baselines.''')
    group.add_argument('-label_smoothing',type=float,default=0.0,
                       help='''Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size -1).
                       Set to zero to turn off label smoothing.''')


    group=parser.add_argument_group('Optimization - Rate')
    group.add_argument('-learning_rate', type=float, default=1.0,
                       help='''Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001''')
    group.add_argument('-learning_rate_decay',type=float,default=0.5,
                       help='''If update_learning_rate, decay learning rate
                       by this if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past start_decay_at.''')
    group.add_argument('-start_decay_at',type=int,default=8,
                       help='''Start decaying every epoch after and including
                       this epoch.''')
    group.add_argument('-start_checkpoint_at',type=int,default=0,
                       help='''Start checkpointing every epoch after and
                       in this epoch.''')
    group.add_argument('-decay_method',type=str,default='',
                       choices=['noam'], help='''Use a custom decay rate.''')
    group.add_argument('-warmup_steps',type=int,default=4000,
                       help='''Number of warmup steps for custom decay.''')


def log_opts(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('-tensorboard', action='store_true',
                       help='''Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.''')
    group.add_argument('-tensorboard_log_dir',type=str,
                       default='exp/tmp',
                       help='''Log directory for Tensorboard.
                       This is also the name of the run.''')
    pass


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)


parser = argparse.ArgumentParser(description='train_multiencoder.py',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

data_opts(parser)
train_opts(parser)
model_opts(parser)
optim_opts(parser)
log_opts(parser)


opt = parser.parse_args()


if opt.layers != -1:
    opt.enc_layers=opt.layers
    opt.dec_layers=opt.layers

if opt.seed>0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print('''WARNING: You have a CUDA device, should run with -gpuid 0.''')

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid)>1:
    sys.stderr.write('''Sorry, multigpu is not supported yet.\n''')
    sys.exit(1)


if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer=SummaryWriter(opt.tensorboard_log_dir + datetime.now().strftime('/%b-%d_%H-%M-%S'),
                         comment='Onmt')



def lazily_load_dataset(corpus_type):
    assert corpus_type in ['train', 'valid']

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s  dataset from %s, number of examples: %d.' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '*pt'))
    assert len(pts) > 0
    for pt in pts:
        yield lazy_dataset_loader(pt, corpus_type)


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.mkdir(model_dirname)


def tally_parameters(model):
    n_params=sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc=0
    dec=0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc+=param.nelement()
        elif 'decoder' or 'generator' in name:
            dec+=param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def build_multiencoder_model(model_opt, opt, fields_dict):
    print('Building Model...')
    model=onmt.ModelConstructorForMultiEncoder.make_multiencoder_model(model_opt, fields_dict,
                                                                       use_gpu(opt))
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model=nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model):
    # Todo: Loading from checkpoint.
    if opt.train_from:
        sys.stderr.write('Sorry, not suppot loading from checkpoint')
        sys.exit(1)
    else:
        print('Making optimizer for training.')
        optim=onmt.Optim(opt.optim,
                         opt.learning_rate, opt.max_grad_norm,
                         lr_decay=opt.learning_rate_decay,
                         start_decay_at=opt.start_decay_at,
                         beta1=opt.adam_beta1,
                         beta2=opt.adam_beta2,
                         adagrad_accum=opt.adagrad_accumulator_init,
                         decay_method=opt.decay_method,
                         warmup_steps=opt.warmup_steps,
                         model_size=opt.rnn_size) # model_size param is used by decay_method 'noam',
                                                  # so we do not use it to some degree.
    optim.set_parameters(model.named_parameters())

    return optim



# Training process related functions.
def report_func(epoch, batch, num_batches,
                progress_step, start_time, lr, report_stats):
    '''
    This is the user-defined batch-level training progress report function.
    :param epoch(int): current epoch count.
    :param batch(int): current batch count.
    :param num_batches(int): total number of batches.
    :param progress_step(int): the progress time.
    :param start_time(float): last report time.
    :param lr(float): current learning rate.
    :param report_stats(Statistics): old Statistics instance.
    :return:
    '''
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard('progress',writer,lr,progress_step)

        report_stats=onmt.Statistics()

    return report_stats


def make_loss_compute(model, tgt_vocab, opt, is_train=True):
    '''
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implemant your
    own loss class, by subclassing LossComputeBase.
    :param model:
    :param tgt_vocab:
    :param opt:
    :param train(bool): train or validate process.
    :return:
    '''
    compute=onmt.Loss.NMTLossCompute(model.generator,
                                     tgt_vocab,label_smoothing=opt.label_smoothing if is_train else 0.0)
    if use_gpu(opt):
        compute.cuda()

    return compute


def make_dataset_iter(dataset_generator,fields, opt, is_train=True):
    '''
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy like
    curriculum learning is ok too.
    :param datasets_generator: generator of dataset (vis torch.load) read from '*.pt' files.
    :param fields: dict of fields(dict of :obj:`Fields`) for different source.
    :param opt:
    :param is_train:
    :return:
    '''
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn=None

    device = opt.gpuid[0] if opt.gpuid else -1 # using 'gpu 0' for allocate data.
    return DatasetLazyIter(dataset_generator,fields,batch_size,batch_size_fn,
                           device,is_train)


class DatasetLazyIter(object):
    '''
    An ordered Dataset Iterator, supporting multiple datasets, and lazy loading.
    Args:
        dataset_generator(generator): a generator/iterator of datasets, which are lazily loaded.
        fields(dict): fields dict for the database.
        batch_size(int):batch size.
        batch_size_fn: custom batch process function.
        device(int): which GPU device to use.
        is_train(bool): train or valid.
    '''
    def __init__(self,dataset_generator,fields, batch_size, batch_size_fn, device, is_train):
        self.dataset_generator=dataset_generator
        self.fields=fields
        self.batch_size=batch_size
        self.batch_size_fn=batch_size_fn
        self.device=device
        self.is_train=is_train

        self.cur_iter=self._next_dataset_iterator(dataset_generator)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter=(d for d in self.dataset_generator)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter=self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # return the len of current dataset.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self,dataset_generator):
        try:
            self.cur_dataset=next(dataset_generator)
        except StopIteration:
            return None

        self.cur_dataset.fields=self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(dataset=self.cur_dataset,batch_size=self.batch_size,
                                       batch_size_fn=self.batch_size_fn,
                                       device=self.device,train=self.is_train,
                                       sort=False,sort_within_batch=True,repeat=False)


def train_model(model, fields, optim, data_type, model_opt):
    assert data_type == 'multi'
    train_loss=make_loss_compute(model,fields['text']['tgt'].vocab,opt)
    valid_loss=make_loss_compute(model,fields['text']['tgt'].vocab,opt,is_train=False)

    trunc_size=opt.truncated_decoder
    shard_size=opt.max_generator_batches
    norm_method=opt.normalization
    grad_accum_count=opt.accum_count # Bool type.

    trainer = TrainerForMultiEncoder(model, train_loss, valid_loss, optim,
                                     trunc_size, shard_size, data_type,
                                     norm_method, grad_accum_count)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs-opt.start_epoch+1,opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs+1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter=make_dataset_iter(lazily_load_dataset('train'),fields, opt)

        train_stats=trainer.train(train_iter, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validae on the validation set.
        valid_iter=make_dataset_iter(lazily_load_dataset('valid'),fields,opt,is_train=False)
        valid_stats=trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.tensorboard: # Todo: tensorboard does not work until now.
            train_stats.log_tensorboard('train',writer,optim.lr,epoch)
            valid_stats.log_tensorboard('valid',writer,optim.lr,epoch)

        # 4. Update the learning rate.
        trainer.epoch_step(valid_stats.ppl(),epoch)

        # 5. Drop a checkpoint if need.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt,epoch,fields,valid_stats)
    # end of for.
    pass



def main():
    # Todo: Load checkpoint if we resume from a previous training.
    if opt.train_from: # opt.train_from defaults 'False'.
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint=torch.load(opt.train_from,
                               map_location=lambda storage, loc: storage)
        model_opt=checkpoint['opt']
        opt.start_epoch=checkpoint['epoch']+1
    else:
        checkpoint=None
        model_opt=opt

    train_dataset = lazily_load_dataset('train')
    ex_generator = next(train_dataset)
    # # {'indices': 0,
    # #  'src': None, # will not be used. should be removed when preparing data.
    # #  'src_audio': FloatTensor,
    # #  'src_path': wav path, # will not be used. should be removed when preparing data.
    # #  'src_text': tuple,
    # #  'tgt': tuple
    # # }
    # For debug.
    # ex=ex_generator[0]
    # getattr(ex,'src_audio',None)
    # getattr(ex,'src_text',None)
    # getattr(ex,'tgt',None)
    pass

    # load vocab
    vocabs = torch.load(opt.data + '.vocab.pt')  # 'src_text', 'tgt'
    vocabs = dict(vocabs)
    pass
    # get fields, we attempt to use dict to store fields for different encoders(source data).
    text_fields = TextDataset.get_fields(0, 0)  # Here we set number of src_features and tgt_features to 0.
                                           # Actually, we can use these features, but it need more modifications.

    audio_fields = AudioDataset.get_fields(0,0)

    # fields['src_text'] = fields['src']  # Copy key from 'src' to 'src_text'. for assigning the field for text type input.
                                        # the field for audio type input will not be made, i.e., fields['src_audio']=audio_fields['src'].
                                        # Because it will not be used next.

    for k, v in vocabs.items():
        v.stoi = defaultdict(lambda: 0, v.stoi)
        if k == 'src_text':
            text_fields['src'].vocab=v
        else:
            text_fields['tgt'].vocab=v
            audio_fields['tgt'].vocab=v


    text_fields = dict([(k, f) for (k, f) in text_fields.items()
                   if k in ex_generator[0].__dict__])  # 'indices', 'src', 'src_text', 'tgt'
    audio_fields=dict([(k,f) for (k,f) in audio_fields.items()
                       if k in ex_generator[0].__dict__])

    print(' * vocabulary size. text source = %d; target = %d' %
          (len(text_fields['src'].vocab), len(text_fields['tgt'].vocab)))
    print(' * vocabulary size. audio target = %d' %
          len(audio_fields['tgt'].vocab))

    fields_dict={'text':text_fields,'audio':audio_fields}
    pass

    # Build model.
    model = build_multiencoder_model(model_opt, opt, fields_dict)  # TODO: support using 'checkpoint'.
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model)  # TODO: support using 'checkpoint'.

    # Do training.
    train_model(model, fields_dict, optim, data_type='multi', model_opt=model_opt)

    # end


if __name__ == '__main__':
    main()
