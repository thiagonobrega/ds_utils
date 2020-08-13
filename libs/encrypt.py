import ngram
import sys

import pandas as pd
from tqdm.notebook import trange, tqdm

sys.path.append(".")
from bloomfilter import BloomFilter
from bloomfilter import dice_coefficient, jaccard_coefficient, entropy_coefficient, overlap_coefficient , hamming_coefficient



def encrypt_ds(df, atts, bf_len, bigrams=2):
    '''
    Criptografa todo o dataset

    :param df: dataset
    :param atts:  atributos e;g; [3,6,8]
    :param bf_len: tamanho do bf (utilizar o get_max_length_of_record())
    :param bigrams: numero de bigramas
    :return: o dataset anonimizado como uma lista de lista [ [id, bf] ]
    '''
    ds = []
    for index, row in df.iterrows():
        id_ = row[0]
        data_ = ''
        for att in atts:
            data_ += row[att]

        e_data = encryptData(data_, bf_len)

        ds.append((id_, e_data))
    return ds

def encryptData(data,size,fp=0.01,bigrams=2,bpower=8,p=None):
    """
        Criptografa um string

        bigrams : 2 = Bigrams
        size : Size of BF
        fp : False positive rate
    """
    bloomfilter = BloomFilter(size,fp,bfpower=bpower)
    if p != None:
        bloomfilter.set_hashfunction_by_p(p)

    index = ngram.NGram(N=bigrams)
    bigrams = list(index.ngrams(index.pad(str(data))))

    for bigram in bigrams:
        bloomfilter.add(str(bigram))

    return bloomfilter

def get_max_length_of_record(df, atts):
    '''
    Informa o tamanho do maior registro de um dataset

    :param df: dataset
    :param atts: a posição dos atributis eg. [1,2,3]
    :return:
    '''
    soma = 0
    for att in atts:
        # .iloc[:,[21,20,22,23,5,7,10,14,13,25,36,30,32,34,36,37,39,27]]
        soma += df.iloc[:, att].map(len).max()
    return soma

def number_of_bigrams(n, r=2):
    '''
    retorna o numero de ngrams para o tamanho n
    :param n: tamanho do string
    :param r: numero de ngrams
    :return:
    '''
    index = ngram.NGram(N=r)
    return len(list(index.ngrams(index.pad(str('a' * n)))))
    #return int(math.factorial(n)/ ( math.factorial(r) * math.factorial(n-r)))

def compare_ds(dfa, dfb, atts, golds, bigrams=2):
    '''
    Compara dois dadataset
    calcula a similaridade de todos os itens de dois dadataset
    metricas [ dice, jaccard, overlap, hamming , entropy ]
    :param dfa:
    :param dfb:
    :param atts: a lista com a posiçãod os atributis

    :param golds: gabarito ja passado pelo metodo datasetutil.gerar_gabarito()
    :param bigrams:

    :return:
    '''
    max_length = max(get_max_length_of_record(dfa, atts), get_max_length_of_record(dfb, atts))

    bf_len = number_of_bigrams(max_length, r=bigrams)

    _dfa = encrypt_ds(dfa, atts, bf_len, bigrams=bigrams)
    _dfb = encrypt_ds(dfb, atts, bf_len, bigrams=bigrams)

    bft = _dfa[0][1]

    # estatisticas
    bf_hash_functions = bft.hash_functions
    bf_bit_size = bft.bit_size
    bf_capacity = bft.capacity

    # comparando tudo
    otodo = []
    for e1 in tqdm(_dfa):
        id1 = e1[0]
        bf1 = e1[1]
        for e2 in _dfb:
            # for e2 in tqdm(eb, leave=False):
            id2 = e2[0]
            bf2 = e2[1]

            dice = dice_coefficient(bf1, bf2)
            jac = jaccard_coefficient(bf1, bf2)
            ol = overlap_coefficient(bf1, bf2)
            ha = hamming_coefficient(bf1, bf2)
            hx = entropy_coefficient(bf1, bf2)

            try:
                classificacao = golds[id1][id2]  # mudar variavel
            except KeyError:
                classificacao = 0  # nao e match

            # linha = {'id1': id1, 'id2': id2,
            #          'dice': dice, 'jaccard': jac, 'overlap': ol,
            #          'hamming': ha,
            #          'entropy': hx,
            #          'is_match': classificacao}
            linha = [id1,id2,dice,jac,ol,ha,hx,classificacao]
            otodo.append(linha)

            # pd.DataFrame(otodo)

    cnames = ['id1', 'id2','dice','jaccard','overlap',
                     'hamming','entropy', 'is_match']

    return pd.DataFrame(otodo,columns=cnames), {'n_hash': bf_hash_functions,'bits': bf_bit_size,'cap': bf_capacity , 'atts' : len(atts)}
    # return pd.DataFrame(otodo), {'n_hash': bf_hash_functions,'bits': bf_bit_size,'cap': bf_capacity , 'atts' : len(atts)}