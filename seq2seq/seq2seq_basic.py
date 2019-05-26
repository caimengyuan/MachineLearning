import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, rnn, Block
from mxnet.contrib import text

from io import open
import collections
import deletime

PAD = '<pad>'       #使每个序列等长
BOS = '<bos>'       #表示序列的开始
EOS = '<eos>'       #表示序列的结束

#定义可以调节的模型参数，在编码器和解码器中分别使用了一层和两层的循环神经网络
epochs = 50
epoch_period = 10

learning_rate = 0.005
max_seq_len = 5         #输入或输出序列的最大长度（含句末添加的EOS字符）

encoder_num_layers = 1      #编码器的隐含层
decoder_num_layers = 2      #解码器的隐含层

encoder_drop_prob = 0.1
decoder_drop_prob = 0.1

encoder_hidden_dim = 256
decoder_hidden_dim = 256
alignment_dim = 25      #公式中的向量V

ctx = mx.cpu(0)

def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []

    with open('fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:
            input_seq, output_seq = line.rstrip().split('\t')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')

            if len(cur_input_tokens) < max_seq_len and len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_output_tokens)
                cur_input_tokens.append(EOS)
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)
                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(PAD)
                output_seqs.append(cur_output_tokens)
        fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens), reseved_tokens = [PAD, BOS, EOS])
        en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens), reserved_tokens = [PAD, BOS, EOS])
    return fr_vocab, en_vocab, input_seqs, output_seqs

if __name__ == '__main__':
    input_vocal, output_vocab, input_seqs ,output_seqs= read_data(max_seq_len)
    X = nd.zeros((len(input_seqs), max_seq_len), ctx = ctx)
    Y = nd.zeros((len(output_seqs), max_seq_len), ctx = ctx)
    for i in range(len(input_seqs)):
        X[i] = nd.array(input_vocal.to_indices(input_seqs[i]), ctx=ctx)
        Y[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)

    dataset = gluon.data.ArrayDataset(X,Y)
