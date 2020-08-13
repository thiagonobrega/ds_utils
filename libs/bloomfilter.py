from __future__ import division
import math
import xxhash
import bitarray
import numpy as np

class BloomFilter:

    # BloomFilter(1000, 0.01)
    # A bloom filter with 1000-element capacity and a 1% false positive rate
    def __init__(self, cap=1000, fpr=0.01, bfpower=8, bs=0):

        self.capacity = cap
        self.false_positive_rate = fpr
        self.error_rate = fpr
        self.number_of_elements = 0
        self.potencia = bfpower

        # Make sure the false positive rate is a reasonable value
        if self.false_positive_rate >= 1 or self.false_positive_rate <= 0:
            raise Exception('Invalid value for false positive rate. Must be in the open interval between 0 and 1.')

        # Calculate the number of bits needed for the false positive rate
        # TODO: need to set a bit_size as a multiple of 8
        if bs == 0:
            n = int(math.ceil(self.capacity * math.log(1 / self.false_positive_rate) / math.log(2) ** 2))
            self.bit_size = round(n / bfpower) * bfpower
        else:
            self.bit_size = bs

        # Make sure the bloom filter is not too large for the 64-bit hash
        # to fill up.
        if self.bit_size > 18446744073709551616:
            raise Exception(
                'BloomFilter is too large for supported hash functions. Make it smaller by reducing capacity or increasing the false positive rate.')
            return

        # Calculate the optimal number of hash functions
        self.hash_functions = int(round(self.bit_size * math.log(2) / self.capacity))

        # Build the empty bloom filter
        self.filter = bitarray.bitarray(self.bit_size)
        self.filter.setall(False)

    def set_hashfunction_by_p(self,p):
        self.hash_functions = abs(int(round(math.log(p) * self.bit_size / self.capacity)))

    # Add an element to the bloom filter
    def add(self, element):
        # Make sure it's not full
        if self.number_of_elements == self.capacity:
            raise Exception('Bloom Filter is full.')
            return

        # Run the hashes and add at the hashed index
        for seed in range(self.hash_functions):
            self.filter[ xxhash.xxh64(element, seed=seed).intdigest() % self.bit_size ] = True
        self.number_of_elements += 1


    # Check the filter for an element
    # False means the element is definitely not in the filter
    # True means the element is PROBABLY in the filter
    def check(self, element):
        # Check each hash location. The seed for each hash is 
        # just incremented from the previous seed starting at 0
        for seed in range(self.hash_functions):
            if not self.filter[ xxhash.xxh64(element, seed=seed).intdigest() % self.bit_size ]:
                return False
        # Probably in the filter if it was at each hashed location
        return True
    
    """
        Implentei intercecao e union
    """
    def intersection(self, other):
        """ Calculates the intersection of the two underlying bitarrays and returns
        a new bloom filter object."""
        if self.capacity != other.capacity or \
            self.error_rate != other.error_rate:
            raise ValueError("Intersecting filters requires both filters to have equal capacity and error rate")
        new_bloom = self.copy()
        new_bloom.filter = new_bloom.filter & other.filter
        return new_bloom
    
    def __and__(self, other):
        return self.intersection(other)
    
    def union(self, other):
        """ Calculates the union of the two underlying bitarrays and returns
        a new bloom filter object."""
        if self.capacity != other.capacity or \
            self.error_rate != other.error_rate:
            raise ValueError("Unioning filters requires both filters to have both the same capacity and error rate")
        new_bloom = self.copy()
        new_bloom.filter = new_bloom.filter | other.filter
        return new_bloom

    def __or__(self, other):
        return self.union(other)

    def copy(self):
        """
            Return a copy of this bloom filter.
        """
        new_filter = BloomFilter(self.capacity, self.error_rate)
        new_filter.filter = self.filter.copy()
        return new_filter

    def __str__(self):
        return str(self.filter.to01())

    # For testing purposes
    def print_stats(self):
        print('Capacity '+str(self.capacity))
        print('Expected Probability of False Positive '+str(self.false_positive_rate))
        print('Bit Size '+str(self.bit_size))
        print('Number of Hash Functions '+str(self.hash_functions))
        print('Number of Elements '+str(self.number_of_elements))

    ##################################################################################
    ###
    ### SPLITING BLOOM FILTER
    ###
    ##################################################################################

    def split(self,n=2,p=256):
        # for i in range(0,n):
        lbs = round(self.bit_size/n)
        cap = round(self.capacity/n)
        a = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=p, bs=lbs)
        b = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=p, bs=lbs)
        a.filter = self.filter[0:lbs]
        b.filter = self.filter[lbs:]
        return a,b


    ##################################################################################
    ###
    ### BLOOM FILTER HARDENING
    ###
    ##################################################################################
    def xor_folding(self):
        """
            Returns a XOR-Folding BloomFilter with one folding

            Schnell, R., Borgs, C., & Encryptions, F. (2016). XOR-Folding for Bloom Encryptions for Record Linkage.
        """
        lbs = round(self.bit_size / 2)
        cap = round(self.capacity / 2)

        fold_pos = round( len(self.filter) / 2 )

        a = self.filter[0:fold_pos]
        b = self.filter[fold_pos:]
        # print('======>',self.bit_size,'#',len(a),len(b))

        r = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=self.potencia, bs=lbs)
        r.filter = a.__ixor__(b)
        return r

    def blip(self,f=0.02):
        """
            BLoom-and-flI (BLIP)

            Schnell, R., & Borgs, C. (2017). Randomized Response and Balanced Bloom Filters for Privacy Preserving Record Linkage.
            IEEE International Conference on Data Mining Workshops, ICDMW, 218–224. https://doi.org/10.1109/ICDMW.2016.0038
        """

        lbs = round(self.bit_size * 1)
        cap = round(self.capacity * 1)


        pf = 0.5 * f
        a = self.filter.copy()

        for i in range(0,len(a)):
            if np.random.random() < pf:
                a[i] = not a[i]


        r = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=self.potencia, bs=lbs)
        r.filter = a
        return r

    def bblip(self,f=0.02):
        """
            Balanced BLoom-and-flI (BBLIP)

            Schnell, R., & Borgs, C. (2017). Randomized Response and Balanced Bloom Filters for Privacy Preserving Record Linkage.
            IEEE International Conference on Data Mining Workshops, ICDMW, 218–224. https://doi.org/10.1109/ICDMW.2016.0038

            The deufault value was chosend by the best result presented by the author
        """
        lbs = round(self.bit_size * 2)
        cap = round(self.capacity * 2)

        a = self.filter.copy()
        b = self.filter.copy()
        b.invert()
        c = a + b

        pf = 0.5 * f


        for i in range(0, len(c)):
            if np.random.random() < pf:
                c[i] = not c[i]

        r = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=self.potencia, bs=lbs)
        r.filter = c
        return r


########
########
########

class NullField():
    """
        Classe utilizada quando o valor  é nulo
    """

    def __init__(self, nome):
        self.nome = nome
        self.sim = 0


def dice_coefficient(filter1, filter2):
    """
        Calculates the overlap coefficient,or Szymkiewicz–Simpson coefficient, of the two underlyng bloom filter.



        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    h = filter1.intersection(filter2).filter.count(True)
    a = filter1.filter.count(True)
    b = filter2.filter.count(True)
    return 2 * h / (a + b)


def jaccard_coefficient(filter1, filter2):
    """
        Calculates the jaccard index between 2 bloom filters

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

        # Brasileiro, i've coded this part using this website [1]
    # Please check this, chunk of code. It may contain error :p
    # 1 -http://blog.demofox.org/2015/02/08/estimating-set-membership-with-a-bloom-filter/

    inter = filter1.intersection(filter2).filter.count(True)
    union = filter1.union(filter2).filter.count(True)

    if union == 0:
        return 0

    return inter / union


def entropy_coefficient(filter1, filter2, base=2):
    """
        Calculates the entropy coeficiente of the two underlyng bloom filter.

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    total_count = int(filter1.bit_size)

    f1_element_count = filter1.filter.count(True)
    f2_element_count = filter2.filter.count(True)

    prob_f1 = f1_element_count / total_count
    prob_f2 = f1_element_count / total_count

    e_f1 = -1.0 * total_count * prob_f1 * math.log(prob_f1) / math.log(base)
    e_f2 = -1.0 * total_count * prob_f2 * math.log(prob_f2) / math.log(base)

    entropy = abs(e_f1 - e_f2)

    #     for element_count in Counter(data).values():
    #         p = element_count / total_count
    #         entropy -= p * math.log(p, self.base)

    assert entropy >= 0

    return 1 - entropy


def overlap_coefficient(filter1, filter2):
    """
        Calculates the Szymkiewicz–Simpson coefficient (or overlap coeficiente) of the two underlyng bloom filter.

        $overlap(X,Y) = \frac{| X \cap Y |}{min(|X|,|Y|)}$
        If set X is a subset of Y or the converse then the overlap coefficient is equal to 1.

         Vijaymeena, M. K.; Kavitha, K. (March 2016). "A Survey on Similarity Measures in Text Mining" (PDF). Machine Learning and Applications: An International Journal. 3 (1): 19–28. doi:10.5121/mlaij.2016.3103.

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    xy = filter1.intersection(filter2).filter.count(True)
    x = filter1.filter.count(True)
    y = filter2.filter.count(True)
    if min(x, y) == 0:
        return 0

    return xy / min(x, y)


def hamming_coefficient(filter1, filter2):
    """
        Calculates the Hamming coeficiente of the two underlyng bloom filter.

        In information theory, the Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different. In other words, it measures the minimum number of substitutions required to change one string into the other, or the minimum number of errors that could have transformed one string into the other.

         Vijaymeena, M. K.; Kavitha, K. (March 2016). "A Survey on Similarity Measures in Text Mining" (PDF). Machine Learning and Applications: An International Journal. 3 (1): 19–28. doi:10.5121/mlaij.2016.3103.

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    xy = filter1.intersection(filter2).filter.count(True)
    x = filter1.filter.count(True)
    y = filter2.filter.count(True)

    changes = min(x - xy, y - xy)
    if changes == 0:
        return 1
    else:
        return 1 - (changes / int(filter1.bit_size))


# The seed should be a string
def test_bloom_filter_performance(number_of_elements = 1000000,false_positive_rate = 0.01, number_of_false_positive_tests = 10000, seed = 'Test'):
    # Let the user know it might take a bit
    print('Performing test, this might take a few seconds.')

    # Make a big bloom filter
    bf = BloomFilter(number_of_elements, false_positive_rate)

    # Fill it right up
    for i in range (number_of_elements):
        bf.add(seed+i)

    # Try things we know aren't in the filter
    false_positives = 0
    for i in range (number_of_false_positive_tests):
        if bf.check(seed+'!'+i):
            false_positives += 1

    # Calculate the tested rate of false positives
    tested_false_positive_rate = false_positives/number_of_false_positive_tests

    # Show the results
    bf.print_stats()
    print('')
    print('Number of False Positive Tests '+number_of_false_positive_tests)
    print('False Positives Detected '+false_positives)
    print('Tested False Positive Rate '+tested_false_positive_rate)

