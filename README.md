# PyKeBABSMachine Learning algorithms already support our daily life in many areas of application.
Be it self-driving cars, intelligent robots, or the field of computational biology. Identi-
fying specific features from biological sequences is essential in understanding the world
surrounding us and its complex connections. Forensics, the development of new drugs,
and tools for predicting certain Illnesses based on our DNA are important subareas of
this Topic. One standard method to do so is in the form of Support Vector Machines
(SVM). SVMs are well suited for this task for several reasons, but the biggest profit
comes from their ability to use the so-called kernels for their calculation. They convert
the alphabetical nature of biological sequences via similarity analysis into their own
numerical similarity Matrices, called Kernel. This Kernel can then be used for internal
calculations. In the case of biological sequences, this special feature clearly distinguishes
the support vector machines from other machine learning algorithms
This thesis is based on the KEBABS Package from Johannes Palme, which was im-
plemented in R. Hence, the user base of R is declining, and the functionalities of the
Kebabs Package are more and more sought after in other Programming languages.
Hence Python is the go-to language for processing biological Data, it makes sense to
provide an easy-to-use framework for manipulating and processing biological Data in
this well-supported platform. The Package offers two Kernels for Kernel-Based Sequence
analysis: the spectrum kernel and the gappy pair Kernel. Both can be used with DNA,
RNA, and Aminoacids sequences. The Package also explicitly represents the calculated
Kernel Matrix as a whole or in the sparse format. The Package also works seamlessly
with multiple existing SVM Frameworks in the Python landscape, like Libsvm and the
prevalent scikit learning library.
The Package also provides Cross-validation, grid search, and an unbiased model se-
lection. For better biological interpretability, the weights used for the calculations can
easily be extracted. Also, prediction profiles have been implemented to understand bet-
ter how some parts of a sequence contribute to the overall result.


