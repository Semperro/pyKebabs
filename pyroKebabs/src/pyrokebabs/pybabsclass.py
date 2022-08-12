import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, precision_score
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from numpy import std
from numpy import mean
from scipy.sparse import coo_matrix, vstack
import seaborn as sns
from collections import deque


class Pybabs:


    sequenceTypes={'dna':0,'rna':1,'aa':2,'aa+s':3}
    #Definition for possible alphabets [DNA/RNA, Amino acids]
    alphabets=['ACGT','ACGU','ACDEFGHIKLMNPQRSTVWY']

    def __init__(self,k,g,data,target):
        self.k = k
        self.g = g
        self.data = data
        self.target = target
        self.dict = {}



    #Assigns a number to a given Input on basis of the used alphabet
    def get_numbers_for_sequence(self,sequence,t=0,reverse=False):
        try:
            ind=[self.alphabets[t].index(x) for x in sequence]
        except ValueError:
            return [-1]
        if reverse:
            rev=[self.alphabets[t].index(x) for x in sequence.reverse_complement()]
            if ind>rev:
                return rev
        return ind

    def _extract_spectrum_sequence(self,sequence, k,t=0,reverse=False):
        """Compute Spectrumkernel for the input with k-mers oif lenght k, by first setting the used alphabet. After a vector (spectrum) for each Sequence is created depending on the alphabet and the value of k,
        the lenght corresponds with the possible combiniations from the two parameters. The multiplier places the values based on their combinations into the vector."""

        n = len(sequence)
        #print ('Seq :' + sequence)
        alphabet=len(self.alphabets[t])
        spectrum = np.zeros(np.power(alphabet, k))
        multiplier = np.power(alphabet, range(k))[::-1]
        #print(multiplier)
        for pos in range(n - k + 1):
            #print(get_numbers_for_sequence(sequence[pos:pos+k]))
            #print ('Mult :')
            #print(multiplier)
            pos_in_spectrum = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+k],t,reverse))

            #print ('Pos :')
            #print(pos_in_spectrum)
            spectrum[pos_in_spectrum] += 1
        return spectrum

    def _extract_gappy_sequence_different(self,sequence, k, g,t=0,reverse=False):
        """Compute gappypair-spectrum for a given sequence, k-mer length k and
        gap length g. A 2*k-mer with a certain gap size is saved at a different
        position than the same 2*k-mer with no gaps or another number of gaps.
        """
        n = len(sequence)
        kk=2*k
        alphabet=len(self.alphabets[t])
        powersize=np.power(alphabet, (kk))
        multiplier = np.power(alphabet, range(kk))[::-1]
        spectrum = np.zeros((g+1)*(powersize))
        for pos in range(n - kk + 1):
                pos_in_spectrum = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+(kk)],t,reverse=reverse))
                spectrum[pos_in_spectrum] += 1
                if (pos+g+kk)<=n:
                    for gap in range(1,g+1):
                        pos_gap = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+k] + sequence[pos+k+gap:pos+gap+kk],t,reverse=reverse))
                        spectrum[(gap*(powersize))+pos_gap] += 1
        return spectrum

    def gappypair_kernel(self,sequences, k, g=0,t=0,sparse=True, reverse=False, include_flanking=False, gapDifferent = True):
        """Compute gappypair-kernel for a set of sequences using k-mer length k
        and gap size g. 
        """
        spectrum = []
        for seq in sequences:
            if include_flanking:
                seq = seq.upper()   
            else:
                seq = seq.upper()
                #print("firstsec: " + seq + '\n')
                seq = Seq("".join([x for x in seq if 'A' <= x <= 'Z']))   
                #print("secondsec: " + seq + '\n')
            if (g>0) and gapDifferent:
                spectrum.append(self._extract_gappy_sequence_different(seq, k, g, t = t, reverse = reverse))
            else:
                #print(t)
                spectrum.append(self._extract_spectrum_sequence(seq, k, t = t, reverse = reverse))
        if sparse:
            return csr_matrix(spectrum)
        #print(spectrum)
        return np.array(spectrum)

    def unbcv(self,X,y,k,g,j=0):
        for i in range(k):
            print('K=' + str(i) + ' G=' + str(j))
            kernel = self.gappypair_kernel(X,k=k, g=g, include_flanking=True, gapDifferent = True, sparse = False)
            print('Kernel for k=' + str(i) + ' and g=' + str(j) + ' built')
            cv_outer = KFold(n_splits=10, shuffle=True)
            outer_results = list()
            for train_ix, test_ix in cv_outer.split(kernel):
                # split data
                X_train, X_test = kernel[train_ix, :], kernel[test_ix, :]
                y_train, y_test = y[train_ix], y[test_ix]
                # configure the cross-validation procedure
                cv_inner = KFold(n_splits=3, shuffle=True)
                # define the model
                model = SVC(kernel='linear')
                # define search space
                space = dict()
                space['C'] = [2**i for i in range(-5, 6)]
                # define search
                search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
                # execute search
                result = search.fit(X_train, y_train)
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)
                # evaluate the model
                acc = accuracy_score(y_test, yhat)
                # store the result
                outer_results.append(acc)
                # report progress
                print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
                # summarize the estimated performance of the model
            print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    

    def unbiasedCV(self,X,y,k,g):
        if g>0:
            for j in range(g):
                self.unbcv(X,y,k,g,j)
        else:
            self.unbcv(X,y,k,g)





    def pybabsSVMtrain(self,data,target,k,g,C):
        kernel = self.gappypair_kernel(data,k=k, g=g,  include_flanking=True, gapDifferent = True, sparse = False)
        X_train, X_test, y_train, y_test = train_test_split(kernel,target,test_size=0.1,random_state=42,stratify=target)
        start = time.time()
        clf = SVC(C=C, kernel='linear')
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test,pred)
        print('acc=' +str(acc))
        print ("Trained linear SVM in {} seconds".format(k, time.time() - start))
        return clf


    def listToString(self,s): 
        # initialize an empty string
        str1 = " " 
        # return string  
        return (str1.join(s))


    def getWeights(self,k,model):
        alphabets=['ACGT']
        perms = []
        permst = list(product(['A','C','G','T'],repeat=k))
        tosort = model.coef_
        done = []
        for i in permst:
            done.append(self.listToString(i))
        df = pd.DataFrame()
        df['letters'] = np.array(done)
        df['val'] = np.array(model.coef_[0])
        df.sort_values(by='val', ascending=False, inplace=True)
        df['letters'] = df['letters'].str.replace(' ', '')
        #dict = df.set_index('letters').T.to_dict(orient='list')
        self.dict = pd.Series(df.val.values,index=df.letters).to_dict()
        return df,self.dict

    def split(self, word):
        return [char for char in word]



    def getPredProfile(self,num,k,data):
        testi = self.split(data[num])
        run = len(data[num])
        pp = np.zeros(run)

        def window(arr, k):
            for i in range(len(arr)-k+1):
                yield arr[i:i+k]
        print('Profile is generated for Seq:' + data[num])
        t=0

        for group in window(data[num], k):
            #if t <4:
                #pp[t] += dict.get(group.upper())/2
            if t < run:
                #print(k)
                #print(t)
                #print(group)
                #print(self.dict.get('ATATG'))
                #print(self.dict.get(group.upper()))
                #print(dict.get(group.upper())/2)
                #print(dict.get(group.upper())/k)
                pp[t:t+k] += self.dict.get(group.upper())/k
                #print(pp)
            t += 1
        #print(pp)
        x = np.arange(0,len(pp))
        y = pp
        sns.set_style("darkgrid")
        sns.set(rc={'figure.figsize':(25,10)})
        t = sns.lineplot(x,y, drawstyle='steps-pre')
        t.set_xticks(range(len(pp)))
        t.set_xticklabels(testi)
        t.set(ylim=(-0.1, 0.1))
        t.axhline(0, color='red')
