import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,GroupKFold,KFold
from sklearn.svm import SVC,LinearSVC
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from itertools import product
import seaborn as sns


class Pybabs:

    #Definition for possible alphabets [DNA/RNA, Amino acids]
    alphabets=['ACGT','ACGU','ACDEFGHIKLMNPQRSTVWY']



    # Constructor
    # Description: Constructor for the PyKebabs object, sets the used alphabet from a string to int and holds the used data
    # @Dependencies     -> None
    # @Functionparams:
    # data              -> contais the data that should be used with the SVM
    # alphbt            -> defines the alphabet "data" is made of
    # @Routineparams:
    # self.data         -> contais the data sued to train the SVM
    # self.target       -> contains the targets for "self.data"
    # self.alphbt       -> stores the sequencetype of "self.data" as integer 
    # @Return           -> None
    def __init__(self,data,alphbt):
        self.data = data['data']
        self.target = data['target']
        if alphbt.upper() == 'DNA':
            self.alphbt = 0
        elif alphbt.upper() == 'RNA':
            self.alphbt=1
        elif alphbt.upper() == 'AA':
            self.alphbt=2
        else:
            print("No alphabet choosen, DNA is used")
            self.alphbt=0
        #make dict a obj var???    
        #self.dict = {}


    # SubRoutine
    # Description: Creates an array containing numbers for a given input on basis of the used alphabet
    # @Dependencies     -> None
    # @Functionparams:
    # sequence          -> holds a single sequence from kernelCreation()
    # reverse           -> defines if the sequences should be read backwards -> for now always False -> for future use
    # @Routineparams:
    # ind               -> contains the index at wich a letter is found in the defined alphabet
    # self.alphbt       -> the alphabet choosen in the pyKeBABS object
    # self.alphabets    -> global array containing the possbile alphabets
    def get_numbers_for_sequence(self,sequence,reverse=False):
        try:
            ind=[self.alphabets[self.alphbt].index(x) for x in sequence]
        except ValueError:
            return [-1]
        if reverse:
            rev=[self.alphabets[self.alphbt].index(x) for x in sequence.reverse_complement()]
            if ind>rev:
                return rev
        return ind

    # SubRoutine
    # Description: Creates a spectrum kernel with kmers of length k
    # @Dependencies     -> import numpy as np
    # @Functionparams:
    # sequence          -> holds a single sequence from kernelCreation()
    # k                 -> contais the k for the kernel-creation
    # reverse           -> defines if the sequences should be read backwards -> for now always False -> for future use
    # @Routineparams:
    # n                 -> number of letters in the sequence
    # self.alphbt       -> the alphabet choosen in the pyKeBABS object
    # self.alphabets    -> global array containing the possbile alphabets
    # alphabet          -> length of the Alphabet choosen in the pyKeBABS object
    # spectrum          -> an array for counting the found patterns
    # multiplier        -> an support array to help calculate the final position of a pattern
    # pos_in_spectrum   -> contains the final position of the actual pattern
    # @Return           -> returns an array containing the pattern counts of the input sequence
    def extract_spectrum_sequence(self,sequence, k,reverse):
        n = len(sequence)
        alphabet=len(self.alphabets[self.alphbt])
        spectrum = np.zeros(np.power(alphabet, k))
        multiplier = np.power(alphabet, range(k))[::-1]
        for pos in range(n - k + 1):
            pos_in_spectrum = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+k],reverse))
            spectrum[pos_in_spectrum] += 1
        return spectrum


    # SubRoutine
    # Description: Creates a gappy pair kernel with kmers of length k and gaps of size g
    # @Dependencies     -> import numpy as np
    # @Functionparams:
    # sequence          -> holds a single sequence from kernelCreation()
    # k                 -> contais the k for the kernel-creation
    # reverse           -> defines if the sequences should be read backwards -> for now always False -> for future use
    # @Routineparams:
    # n                 -> number of letters in the sequence
    # kk                -> Equals to k squared
    # self.alphbt       -> the alphabet choosen in the pyKeBABS object
    # self.alphabets    -> global array containing the possbile alphabets
    # alphabet          -> length of the Alphabet choosen in the pyKeBABS object
    # powersize         -> defines the exponent used in calculation depnding on the alphabet
    # spectrum          -> an array for counting the found patterns
    # multiplier        -> an support array to help calculate the final position of a pattern
    # pos_in_spectrum   -> contains the final position of the actual pattern
    # @Return           -> returns an array containing the pattern counts of the input sequence
    def extract_gappy_seqeunce(self,sequence, k, g,reverse=False):
        #print("gappy comp")
        n = len(sequence)
        kk=2*k
        alphabet=len(self.alphabets[self.alphbt])
        powersize=np.power(alphabet, (kk))
        multiplier = np.power(alphabet, range(kk))[::-1]
        spectrum = np.zeros((g+1)*(powersize))
        for pos in range(n - kk + 1):
                pos_in_spectrum = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+(kk)],reverse=reverse))
                spectrum[pos_in_spectrum] += 1
                if (pos+g+kk)<=n:
                    for gap in range(1,g+1):
                        pos_gap = np.sum(multiplier * self.get_numbers_for_sequence(sequence[pos:pos+k] + sequence[pos+k+gap:pos+gap+kk],reverse=reverse))
                        spectrum[(gap*(powersize))+pos_gap] += 1
        return spectrum



    # SubRoutine
    # Description: Decides (g>0) if a gappy or spectrum kernel is created.
    # @Dependencies     -> from scipy.sparse import csr_matrix
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # reverse           -> defines if the sequences should be read backwards -> for now always False -> for future use
    # include_flanking  -> defines if lower letters are considered -> for now always True -> for future use
    # @Routineparams:
    # self.data         -> contains the seuences used as training
    # spectrum          -> hold the final kernel object
    # seq               -> holds a single sequence
    # @Return           -> returns the choosen kernel
    def kernelCreation(self, k, g=0, reverse=False, include_flanking=True):
        spectrum = []
        for seq in self.data:
            if include_flanking:
                seq = seq.upper()   
            if (g>0):
                spectrum.append(self.extract_gappy_seqeunce(seq, k, g, reverse = reverse))
            else:
                spectrum.append(self.extract_spectrum_sequence(seq, k, reverse = reverse))
        return np.array(spectrum)

    # SubRoutine
    # Description: Creates the gram-version of the kernel
    # @Dependencies     -> from scipy.sparse import csr_matrix
    # @Functionparams:
    # kernel            -> the kernel that should be converted
    # @Routineparams:
    # traingram         -> holds the converted gram_kernel
    # @Return           -> returns the gram-kernel
    def createGram(self,kernel):
        traingram = np.dot(kernel,kernel.T)
        return traingram


    # SubRoutine
    # Description: Creates the final kernel object regarding to the user inputs
    # @Dependencies     -> from scipy.sparse import csr_matrix
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # @Routineparams:
    # kernel            -> always hold the kernel used for calculation
    # row_sums          -> hold the amount of rows int the kernel
    # @Return           -> returns the kernel with given adaptions
    def createKernel(self,k,g,sparse,gram,norm):
        kernel = self.kernelCreation(k, g)
        if norm:
            row_sums = kernel.sum(axis=1)
            kernel = kernel/row_sums[:, np.newaxis]
        if sparse:
            kernel = csr_matrix(kernel)
        if gram == True:
                self.createGram(kernel)
        return kernel



    # SubRoutine
    # Description: Performs a nested Crossvalidation
    # @Dependencies     -> from sklearn.model_selection import GridSearchCV
    #                   -> from sklearn.svm import LinearSVC
    #                   -> from sklearn.model_selection import KFold
    #                   -> from sklearn.svm import SVC
    #                   -> from sklearn.metrics import accuracy_score
    #                   -> from numpy import std
    #                   -> from numpy import mean
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # outercv           -> defines how much cross-validations are run in the outer loop
    # innercv           -> defines how much cross-validations are run in the inner nested loop
    # verbose           -> sets the amount of information provided to the console
    # C                 -> defines the C-Costfactor for the SVM
    # svmc              -> defines which SVM implementation is used for calculation
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # @Routineparams:
    # kernel            -> always hold the kernel used for calculation
    # cv_outer          -> crossvalidation object defining the outer loop
    # outer_results     -> holds the accuracys for the results of the outer loop
    # model             -> defines the SVM-model used for calculations
    # train_ix,test_ix  -> hold the test/train data in the form of the kernel
    # X_train,X_test    -> hold the test/train data that was split again from the outer split
    # y_train, y_test   -> hold the target vector for the test/train sets
    # cv_inner          -> crossvalidation object defining the inner loop
    # space             -> contains the hyperparameters that should be tested in the inner loop
    # search            -> gridSearch object for the inner loop
    # result            -> holds the models created with @search
    # best_model        -> holds the best model found in the inner loop
    # yhat              -> holds the test results for the best model
    # acc               -> holds the test accuracy of the best model
    # @Return           -> returns the best model and the mean accuracy above all models
    def unbcv(self,k,g,outercv,innercv,verbose,C,svmc,sparse,gram,norm):
            kernel = self.createKernel(k,g,sparse,gram,norm)
            cv_outer = KFold(n_splits=outercv, shuffle=False)
            outer_results = list()
            # define the model
            if svmc.upper()=='LIB':
                print('Liblinear was choosen')
                model = LinearSVC()
            else:           
                print('SVC was choosen')
                model = SVC(kernel='linear')
            for train_ix, test_ix in cv_outer.split(kernel):
                X_train, X_test = kernel[train_ix, :], kernel[test_ix, :]
                y_train, y_test = self.target[train_ix], self.target[test_ix]
                cv_inner = KFold(n_splits=innercv, shuffle=False)
                space = dict()
                space['C'] = C
                search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True, verbose=verbose-2)
                result = search.fit(X_train, y_train)
                best_model = result.best_estimator_
                yhat = best_model.predict(X_test)
                acc = accuracy_score(y_test, yhat)
                outer_results.append(acc)
                if verbose >= 2:
                    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
            if verbose >= 1:    
                print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))  
            return best_model,np.mean(outer_results),k,g






    # SubRoutine
    # Description: Creates loops to find the best kernel hyperparameters using unbcv()
    # @Dependencies     -> from numpy import mean
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # outercv           -> defines how much cross-validations are run in the outer loop
    # innercv           -> defines how much cross-validations are run in the inner nested loop
    # verbose           -> sets the amount of information provided to the console
    # svmc              -> defines which SVM implementation is used for calculation
    # C                 -> defines the C-Costfactor for the SVM
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # best_model        -> holds the current best found model
    # best_acc          -> holds the best current accuracy
    # @Routineparams:
    # model             -> holds the best model found in unbcv()
    # @Return           -> returns the best model for the actual Kernel configuration,as well as the coresponding accuracy, and the K/G where it was found
    def kerneloptimisation(self,k,outercv,innercv,verbose,C,svmc,sparse,gram,norm,best_model=(0,0),best_acc=0,j=0):
        for i in range(1,k+1):
            if verbose >=1:
                print('K=' + str(i) + ' G=' + str(j))
                print('Kernel for k=' + str(i) + ' and g=' + str(j) + ' built')
            model = self.unbcv(i,j,outercv,innercv,verbose,C,svmc,sparse,gram,norm)
            if model[1] > best_acc:
                print("Accuracy new best model:" , np.mean(model[1]))
                best_acc = np.mean(model[1])
                best_model = model[0]
            else:
                print("Modell has worse key figures")
        return best_model,best_acc,i,j


    # Routine
    # Description: Creates loops to find the best kernel hyperparameters using unbcv()
    # @Dependencies     -> from numpy import mean
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # kernopt           -> defines if the kernel hyperparameters are optimized or not
    # outercv           -> defines how much cross-validations are run in the outer loop
    # innercv           -> defines how much cross-validations are run in the inner nested loop
    # svmc              -> defines which SVM implementation is used for calculation
    # C                 -> defines the C-Costfactor for the SVM , must be an array no int
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # verbose           -> sets the amount of information provided to the console
    # @Routineparams:
    # best_model        -> holds the current best found model
    # @Return           -> returns the best model for the all Kernel configuration,as well as the coresponding accuracy, and the K/G where it was found
    def unbiasedCV(self,k,g,kernopt=False,outercv=6,innercv=3,svmc='SVM',sparse=False,gram=False,norm=False,C=[2**i for i in range(-1, 3)],verbose=0):
        if kernopt == True:
            if g > 0:
                best_model = (0,0)
                for j in range(1,g+1):
                    best_model = self.kerneloptimisation(k,outercv,innercv,verbose,C,svmc,sparse,gram,norm,best_model[0],best_model[1],j)
                return best_model
            else:
                best_model = self.kerneloptimisation(k,outercv,innercv,verbose,C,svmc,sparse,gram,norm)
                return best_model
        else:
            best_model = self.unbcv(k,g,outercv,innercv,verbose,C,svmc,sparse,gram,norm)
            return best_model




    # Routine
    # Description: Performs grouped cross-validatoin
    # @Dependencies     -> from sklearn.model_selection import GridSearchCV
    #                   -> from sklearn.svm import LinearSVC
    #                   -> from sklearn.model_selection import GroupKFold
    #                   -> from sklearn.svm import SVC
    #                   -> from sklearn.metrics import accuracy_score
    #                   -> from numpy import std
    #                   -> from numpy import mean
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # groups            -> contains the groups in which the data is divided
    # outercv           -> defines how much cross-validations are run in the outer loop
    # innercv           -> defines how much cross-validations are run in the inner nested loop
    # svmc              -> defines which SVM implementation is used for calculation
    # C                 -> defines the C-Costfactor for the SVM , must be an array no int
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # verbose           -> sets the amount of information provided to the console
    # @Routineparams:
    # kernel            -> always hold the kernel used for calculation
    # group_kfold       -> crossvalidation object defining the outer loop
    # outer_results     -> holds the accuracys for the results of the outer loop
    # model             -> defines the SVM-model used for calculations
    # train_ix,test_ix  -> hold the test/train data in the form of the kernel
    # X_train,X_test    -> hold the test/train data that was split again from the outer split
    # y_train, y_test   -> hold the target vector for the test/train sets
    # cv_inner          -> crossvalidation object defining the inner loop
    # space             -> contains the hyperparameters that should be tested in the inner loop
    # search            -> gridSearch object for the inner loop
    # result            -> holds the models created with @search
    # best_model        -> holds the best model found in the inner loop
    # yhat              -> holds the test results for the best model
    # acc               -> holds the test accuracy of the best model
    # @Return           -> returns the best model and the mean accuracy above all models
    def groupedCV(self,k,g,groups,svmc='SVM',sparse=False,gram=False,norm=False,verbose=2,outercv=6,innercv=3,C=[2**i for i in range(-1, 3)]):
        kernel = self.createKernel(k,g,sparse,gram,norm)
        if outercv > len(np.unique(groups)):
            outercv = len(np.unique(groups))
        group_kfold = GroupKFold(n_splits=outercv)
        i=0
        outer_results = list()
            # define the model
        if svmc.upper()=='LIB':
            print('Liblinear was choosen')
            model = LinearSVC()
        else:           
            print('SVC was choosen')
            model = SVC(kernel='linear')

        for train_ix, test_ix in group_kfold.split(X=kernel,y=self.target,groups=groups):
            X_train, X_test = kernel[train_ix, :], kernel[test_ix, :]
            y_train, y_test = self.target[train_ix], self.target[test_ix]
            if verbose >= 3:
                print(f"TRAIN INDEXES: {train_ix}, TEST INDEXES: {test_ix}\n")
            cv_inner = KFold(n_splits=innercv, shuffle=False)
            space = dict()
            space['C'] = C
            search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True, verbose=verbose-2)
            result = search.fit(X_train, y_train)
            best_model = result.best_estimator_
            yhat = best_model.predict(X_test)
            acc = accuracy_score(y_test, yhat)
            outer_results.append(acc)
            if verbose >= 2:
                    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        if verbose >= 1:    
            print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))  
        return best_model,np.mean(outer_results)



    # Routine
    # Description: Trains a SVM with giver parameters
    # @Dependencies     -> from sklearn.svm import LinearSVC
    #                   -> from sklearn.svm import SVC
    # @Functionparams:
    # k                 -> contais the k for the kernel-creation
    # g                 -> contais the g for the kernel-creation
    # svmc              -> defines which SVM implementation is used for calculation
    # C                 -> defines the C-Costfactor for the SVM , must be an array no int
    # sparse            -> defines if a sparse kernel is used
    # gram              -> defines if the gram-Matrix is used
    # norm              -> defines if the kernel used should be normalized
    # @Routineparams:
    # kernel            -> always hold the kernel used for calculation
    # model             -> defines the SVM-model used for calculations
    # @Return           -> returns the SVM model trained with the given parameters,the acc as 0, and g/k
    def pybabsSVMtrain(self,k,g,C,svmc="SVC",sparse=False,gram=False,norm=False):
        if svmc.upper()=='LIB':
            print('Liblinear was choosen')
            model = LinearSVC(C=C)
        else:           
            print('SVC was choosen')
            model = SVC(kernel='linear',C=C)
        kernel = self.createKernel(k,g,sparse,gram,norm)
        clf = model
        clf.fit(kernel, self.target)
        return (clf,0,k,g)


    # SubRoutine
    # Description: Converts a list of string in one string
    # @Dependencies     -> None
    # @Functionparams:
    # s                 -> holds the incoming list
    # @Routineparams:
    # str1              -> holds the final string
    # @Return           -> return a single string
    def listToString(self,s): 
        str1 = " " 
        return (str1.join(s))



    # Routine
    # Description: Extracts the weight of a given model
    # @Dependencies     -> import numpy as np
    #                   -> import pandas as pd
    # @Functionparams:
    # model             -> contains the model of wich the weights should be extracted
    # @Routineparams:
    # self.alphbt       -> the alphabet choosen in the pyKeBABS object
    # perms             -> a list cointaining every possible pattern of the sequencetype
    # dict              -> a dictinary that hold the pattern names as well as the coresponding values
    # df                -> a pandas dataframe that hold the pattern names as well as the coresponding values for visualization
    # @Return           -> returns a dictionary and a pandas dataframe with the same content, df for visualization, dict for further use in getPredProfile()
    def getWeights(self,model):
        if self.alphbt == 1:
            perms = list(product(['A','C','G','U'],repeat=model[2]))
        elif self.alphbt == 2:
            perms = list(product(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],repeat=model[2]))
        else:
            perms = list(product(['A','C','G','T'],repeat=model[2]))
        dict = {}
        df = pd.DataFrame()
        patterns = []
        for i in perms:
            patterns.append(self.listToString(i)) 
        df['letters'] = np.array(patterns)
        df['letters'] = df['letters'].str.replace(' ', '')
        #print(len(model.coef_[0]))
        #print(model[0].coef_[0])
        df['val'] = np.array(model[0].coef_[0])
        df.sort_values(by='val', ascending=False, inplace=True)   
        dict = pd.Series(df.val.values,index=df.letters).to_dict()
        return df,dict

    # SubRoutine
    # Description: splits a string into an array with every char on an index
    # @Dependencies     -> None
    # @Functionparams:
    # word              -> contains the string to split
    # @Routineparams:
    # @Return           -> returns an array with every char of @word on an index
    def split(self, word):
        return [char for char in word]


    # Routine
    # Description: Extracts the prediction-profile of a given model by taking the next k indices - sum them up and then divide by k
    # @Dependencies     -> import numpy as np
    #                   -> import pandas as pd
    # @Functionparams:
    # model             -> contains the model of wich the prediction-profile should be extracted
    # num               -> contains the number of the sequence that should be displayed
    # @Routineparams:
    # legendletters     -> holds the labels for the x-axis
    # run               -> an int as high as the number of letters in the sequence
    # predProf          -> an array that holds the values calculated for the prediction-profile
    # self.data         -> the unsplit trainings data
    # t                 -> run variable for the length of the sequence
    # dict              -> a dictinary that hold the pattern names as well as the coresponding weight-values
    # x                 -> x values for plot
    # y                 -> y values for plot
    # ppl               -> list of the prediction-profile weights
    # low               -> varible limit for the y height depending in highest vlaue in @ppl
    # high              -> varible limit for the y height depending in highest vlaue in @ppl
    # @Return           -> returns a dictionary and a pandas dataframe with the same content, df for visualization, dict for further use in getPredProfile()
    def getPredProfile(self,model,num):
        legendletters = self.split(self.data[num])
        run = len(self.data[num])
        predProf = np.zeros(run)
        print("Prediction profiles are generated with k =" , model[2] )
        def window(self):
            for i in range(len(self.data[num])-model[2]+1):
                yield self.data[num][i:i+model[2]]
        print('Profile is generated for Seq:' + self.data[num])
        t=0
        df, dict = self.getWeights(model)
        for group in window(self):
            if t < run:
                predProf[t:t+4] += dict.get(group.upper())/model[2]
            t += 1
        x = np.arange(0,len(predProf))
        y = predProf
        sns.set(font_scale=3)
        sns.set_style("darkgrid")
        plt.figure(figsize=(30, 12))
        t = sns.lineplot(x=x,y=y, drawstyle='steps-pre', label=num+1)
        t.set_xticks(range(len(predProf)))
        t.set_xticklabels(legendletters)
        ppl = predProf.tolist()
        low = (min(ppl))*-1
        high =  max(ppl)
        if high > (low):
            t.set(ylim=(-high-high/10, high+high/10))
        else:
            t.set(ylim=(-low-low/10, low+low/10))
        t.axhline(0, color='red')
