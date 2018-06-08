#!/usr/bin/env python

from __future__ import division
from collections import defaultdict

import argparse
import glob
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import onmt
import onmt.ModelConstructorForMultiEncoder

from onmt.io.TextDataset import TextDataset
from onmt.Utils import use_gpu


def train_opts(parser):
    parser.add_argument('-data', required=True,
                        help='''Path to input data.''')
    parser.add_argument('-save_model', required=True,
                        help='''Path to save model.''')
    pass


def model_opts(parser):
    pass


parser = argparse.ArgumentParser(description='train_multiencoder.py',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
train_opts(parser)
model_opts(parser)

opt = parser.parse_args()


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
    for name, param in model.parameters():
        if 'encoder' in name:
            enc+=param.nelement()
        elif 'decoder' or 'generator' in name:
            dec+=param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def build_multiencoder_model(model_opt, opt, fields):
    print('Building Model...')
    model=onmt.ModelConstructorForMultiEncoder.make_multiencoder_model(model_opt, fields,
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
    :param fields:
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
    train_loss=make_loss_compute(model,fields['tgt'].vocab,opt)
    valid_loss=make_loss_compute(model,fields['tgt'].vocab,opt,is_train=False)

    trunc_size=opt.truncated_decoder
    shard_size=opt.max_generator_batches
    norm_method=opt.normalization
    grad_accum_count=opt.accum_count # Boolean type.

    trainer = onmt.TrainerForMultiEncoder(model, train_loss, valid_loss, optim,
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
    # get fields
    fields = TextDataset.get_fields(0, 0)  # Here we set number of src_features and tgt_features to 0.
                                           # Actually, we can use these features, but it need more modifications.

    fields['src_text'] = fields['src']  # Copy key from 'src' to 'src_text'. for assigning the field for text type input.
                                        # the field for audio type input will not be made, i.e., fields['src_audio']=audio_fields['src'].
                                        # Because it will not be used next.
    for k, v in vocabs.items():
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v

    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in ex_generator[0].__dict__])  # 'indices', 'src', 'src_text', 'tgt'

    print(' * vocabulary size. text source = %d; target = %d' %
          (len(fields['src_text'].vocab), len(fields['tgt'].vocab)))
    pass

    # Build model.
    model = build_multiencoder_model(model_opt, opt, fields)  # TODO: support using 'checkpoint'.
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model)  # TODO: support using 'checkpoint'.

    # Do training.
    train_model(model, fields, optim, data_type='multi', model_opt=model_opt)

    # end


if __name__ == '__main__':
    main()
