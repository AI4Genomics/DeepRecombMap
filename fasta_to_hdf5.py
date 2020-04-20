#!/usr/bin/env python
from __future__ import print_function
import sys
from collections import OrderedDict

import h5py
import numpy as np
import numpy.random as npr
from sklearn import preprocessing

import pandas as pd

from optparse import OptionParser

################################################################################
# seq_hdf5.py
#
# Make an HDF5 file for Torch input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='add_features_file', default=None, help='Table of additional features')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-c', dest='counts', default=False, action='store_true', help='Validation and training proportions are given as raw counts [Default: %default]')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_pct', default=0, type='float', help='Test % [Default: %default]')
    parser.add_option('-v', dest='valid_pct', default=0, type='float', help='Validation % [Default: %default]')
    parser.add_option('--vt', dest='valid_test', default=False, action='store_true', help='Use validation as test, too [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        out_file = args[2]

    # seed rng before shuffle
    npr.seed(options.random_seed)

################################################################################
# align_seqs_scores
#
# Align entries from input dicts into numpy matrices ready for analysis.
#
# Input
#  seq_vecs:      Dict mapping headers to sequence vectors.
#  seq_scores:    Dict mapping headers to score vectors.
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def align_seqs_scores_1hot(seq_vecs, sort=True):
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()


    # construct lists of vectors
    #train_scores = []
    train_seqs = []
    for header in seq_headers:
        if (header in seq_vecs):
            train_seqs.append(seq_vecs[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    #train_scores = np.vstack(train_scores)

    return train_seqs #train_scores



################################################################################
# dna_one_hot
#
# Input
#  seq:
#
# Output
#  seq_vec: Flattened column vector
################################################################################
'''
def dna_one_hot(seq, seq_len=None):
    if seq_len == None:
        seq_len = len(seq)

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    seq_code = np.zeros((4,seq_len), dtype='int8')
    for i in range(seq_len):
        try:
            seq_code[int(seq[i]),i] = 1
        except:
            # print >> sys.stderr, 'Non-ACGT nucleotide encountered'
            seq_code[:,i] = 0.25

    # flatten and make a column vector 1 x len(seq)
    seq_vec = seq_code.flatten()[None,:]

    return seq_vec
'''
def dna_one_hot(seq, seq_len=None, flatten=True):
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len) // 2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[int(seq[i-seq_start]),i] = 1
            except:
                seq_code[:,i] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None,:]

    return seq_vec




################################################################################
# hash_sequences_1hot
#
# Input
#  fasta_file:  Input FASTA file.
#  extend_len:  Extend the sequences to this length.
#
# Output
#  seq_vecs:    Dict mapping FASTA headers to sequence representation vectors.
################################################################################

def hash_sequences_1hot(fasta_file, extend_len=None):
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = 0
        seq = ''
        for line in open(fasta_file):
            if line[0] == '>':
                if seq:
                    seq_len = max(seq_len, len(seq))

                header = line[1:].rstrip()
                seq = ''
            else:
                seq += line.rstrip()

        if seq:
            seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ''
    for line in open(fasta_file):
        if line[0] == '>':
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ''
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs


################################################################################
# load_data_1hot
#
# Input
#  fasta_file:  Input FASTA file.
#  scores_file: Input scores file.
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def load_data_1hot(fasta_file, extend_len=None, mean_norm=True, whiten=False, permute=True, sort=False):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file)

    train_seqs = align_seqs_scores_1hot(seq_vecs, sort)

    '''# whiten scores
    if whiten:
        train_scores = preprocessing.scale(train_scores)
    elif mean_norm:
        train_scores -= np.mean(train_scores, axis=0)'''

    # randomly permute
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]
        #train_scores = train_scores[order]

    return train_seqs #train_scores

    #################################################################
    # load data
    #################################################################
    seqs, targets = load_data_1hot(fasta_file, targets_file, mean_norm=False, whiten=False, permute=False, sort=False)

    # reshape sequences for torch
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))

    # read headers
    headers = []
    for line in open(fasta_file):
        if line[0] == '>':
            headers.append(line[1:].rstrip())
    headers = np.array(headers)

    # read labels
    target_labels = open(targets_file).readline().strip().split('\t')

    # read additional features
    if options.add_features_file:
        df_add = pd.read_table(options.add_features_file, index_col=0)
        df_add = df_add.astype(np.float32, copy=False)

    # permute
    if options.permute:
        order = npr.permutation(seqs.shape[0])
        seqs = seqs[order]
        targets = targets[order]
        headers = headers[order]
        if options.add_features_file:
            df_add = df_add.iloc[order]

    # check proper sum
    if options.counts:
        assert(options.test_pct + options.valid_pct <= seqs.shape[0])
    else:
        assert(options.test_pct + options.valid_pct <= 1.0)


  #################################################################
    # divide data
    #################################################################
    if options.counts:
        test_count = int(options.test_pct)
        valid_count = int(options.valid_pct)
    else:
        test_count = int(0.5 + options.test_pct * seqs.shape[0])
        valid_count = int(0.5 + options.valid_pct * seqs.shape[0])

    train_count = seqs.shape[0] - test_count - valid_count
    train_count = batch_round(train_count, options.batch_size)
    print('%d training sequences ' % train_count, file=sys.stderr)

    test_count = batch_round(test_count, options.batch_size)
    print('%d test sequences ' % test_count, file=sys.stderr)

    valid_count = batch_round(valid_count, options.batch_size)
    print('%d validation sequences ' % valid_count, file=sys.stderr)

    i = 0
    train_seqs, train_targets = seqs[i:i+train_count,:], targets[i:i+train_count,:]
    i += train_count
    valid_seqs, valid_targets, valid_headers = seqs[i:i+valid_count,:], targets[i:i+valid_count,:], headers[i:i+valid_count]
    i += valid_count
    test_seqs, test_targets, test_headers = seqs[i:i+test_count,:], targets[i:i+test_count,:], headers[i:i+test_count]

    if options.add_features_file:
        i = 0
        train_add = df_add.iloc[i:i+train_count]
        i += train_count
        valid_add = df_add.iloc[i:i+valid_count]
        i += valid_count
        test_add = df_add.iloc[i:i+test_count]



    #################################################################
    # construct hdf5 representation
    #################################################################
    h5f = h5py.File(out_file, 'w')

    h5f.create_dataset('target_labels', data=target_labels)

    if train_count > 0:
        h5f.create_dataset('train_in', data=train_seqs)
        h5f.create_dataset('train_out', data=train_targets)

    if valid_count > 0:
        h5f.create_dataset('valid_in', data=valid_seqs)
        h5f.create_dataset('valid_out', data=valid_targets)

    if test_count > 0:
        h5f.create_dataset('test_in', data=test_seqs)
        h5f.create_dataset('test_out', data=test_targets)
        h5f.create_dataset('test_headers', data=test_headers)
    elif options.valid_test:
        h5f.create_dataset('test_in', data=valid_seqs)
        h5f.create_dataset('test_out', data=valid_targets)
        h5f.create_dataset('test_headers', data=valid_headers)

    if options.add_features_file:
        h5f.create_dataset('add_labels', data=list(df_add.columns))

        if train_count > 0:
            h5f.create_dataset('train_add', data=train_add.as_matrix())
        if valid_count > 0:
            h5f.create_dataset('valid_add', data=valid_add.as_matrix())
        if test_count > 0:
            h5f.create_dataset('test_add', data=test_add.as_matrix())
        elif options.valid_test:
            h5f.create_dataset('test_add', data=valid_add.as_matrix())

    h5f.close()


def batch_round(count, batch_size):
    if batch_size != None:
        count -= (batch_size % count)
    return count

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
