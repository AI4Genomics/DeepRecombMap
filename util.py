import datetime
import numpy as np


def cal_iter_time(former_iteration_endpoint, tz):
    """Calculating 'Computation Time' for this round of iteration"""
    current_time = datetime.datetime.now(tz)
    time_elapsed = current_time - former_iteration_endpoint
    time_elapsed = str(time_elapsed).split(".")[0]
    #print(" ~~  Time current: {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
    #print("~~~ Time elapsed for this epoch: {} ~~~\n\n".format(str(time_elapsed).split(".")[0]))
    return current_time, time_elapsed


#TODO: make sure mapping is the same order used in the preprocessing steps
def dna_to_onehot(seq_nts):
    """function that takes a dna sequence of nucleotides and returns onehot-encoded representation of it"""
    nt_dict = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]
              }
    return np.asarray([nt_dict[nt.upper()] for nt in seq_nts]).transpose() # transpose to have nb_cols=nb_nts


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)
    
#TODO: transform one-hot array to dna characters (not vector)
"""def vecs2dna(seq_vecs):
    '''
    Input:
        seq_vecs:
    Output:
        seqs
    '''

    # possibly reshape
    if len(seq_vecs.shape) == 2:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))
    elif len(seq_vecs.shape) == 4:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))

    seqs = []
    for i in range(seq_vecs.shape[0]):
        seq_list = ['']*seq_vecs.shape[2]
        for j in range(seq_vecs.shape[2]):
            if seq_vecs[i,0,j] == 1:
                seq_list[j] = 'A'
            elif seq_vecs[i,1,j] == 1:
                seq_list[j] = 'C'
            elif seq_vecs[i,2,j] == 1:
                seq_list[j] = 'G'
            elif seq_vecs[i,3,j] == 1:
                seq_list[j] = 'T'
            elif seq_vecs[i,:,j].sum() == 1:
                seq_list[j] = 'N'
            else:
                print('Malformed position vector: ', seq_vecs[i,:,j], 'for sequence %d position %d' % (i,j), file=sys.stderr)
        seqs.append(''.join(seq_list))
    return seqs

#test
seq = "ATGCAATTAGCT"
seq_vec = dna_to_onehot(seq)
vecs2dna(seq_vec)"""
