```python
# Based on this path-breaking paper: 
# http://wordvec.colorado.edu/papers/Landauer_Foltz_Laham_1998.pdf

import string
import re
import nltk
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.plotting import scatterplotmatrix

nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package punkt to /home/test/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /home/test/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
#In the command below we are creating our sample dataset 
# that we will use in this lab. This is a text dataset in the raw form.

Sentences = ["Human machine interface for ABC computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The generation of random, binary, ordered trees", "The intersection graph of paths in trees", "Graph minors IV: Widths of trees and well-quasi-ordering", "Graph minors: A survey"]
print(Sentences)
```

    ['Human machine interface for ABC computer applications', 'A survey of user opinion of computer system response time', 'The EPS user interface management system', 'System and human system engineering testing of EPS', 'Relation of user perceived response time to error measurement', 'The generation of random, binary, ordered trees', 'The intersection graph of paths in trees', 'Graph minors IV: Widths of trees and well-quasi-ordering', 'Graph minors: A survey']



```python
# Format the data to be worked on

#Let's lowercase each word in the list above
NewSentences = []
for OneSentence in Sentences:
  OneSentence = OneSentence.lower()
  NewSentences.append(OneSentence)

#Next step is to remove stop words from text. 
# This helps reduce the dimensions of the matrix that we create.
stopwords_dict = {word: 1 for word in stopwords.words("english")}
for OneSentence in Sentences:
  OneSentence = " ".join([word for word in OneSentence.split() if word not in stopwords_dict])
  NewSentences.append(OneSentence)
    
#Now, we remove punctuations from string
for word in Sentences:
    for character in word:
        if character in string.punctuation:
            word = word.replace(character,"")
    NewSentences.append(word)

#The code above has a problem. 
#Please look carefully and let me know where the problems are. 
#We must fix the problems before moving ahead. 
#Besides that, how will you ensure that the fixed text is used in your future runs.
for word in Sentences:
    for character in word:
        if character in string.punctuation:
            word = re.sub('[^a-zA-Z0-9]+\s*', ' ', word)
    NewSentences.append(word)

#A common step in natural language processing is tokenising the string 
#(https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization).
Sentences = [word_tokenize(i) for i in Sentences]
    
print(Sentences)
```

    [['Human', 'machine', 'interface', 'for', 'ABC', 'computer', 'applications'], ['A', 'survey', 'of', 'user', 'opinion', 'of', 'computer', 'system', 'response', 'time'], ['The', 'EPS', 'user', 'interface', 'management', 'system'], ['System', 'and', 'human', 'system', 'engineering', 'testing', 'of', 'EPS'], ['Relation', 'of', 'user', 'perceived', 'response', 'time', 'to', 'error', 'measurement'], ['The', 'generation', 'of', 'random', ',', 'binary', ',', 'ordered', 'trees'], ['The', 'intersection', 'graph', 'of', 'paths', 'in', 'trees'], ['Graph', 'minors', 'IV', ':', 'Widths', 'of', 'trees', 'and', 'well-quasi-ordering'], ['Graph', 'minors', ':', 'A', 'survey']]



```python
#Create a term document matrix: 
#https://en.wikipedia.org/wiki/Document-term_matrix
mlb = MultiLabelBinarizer()
TermDocumentMatrix = mlb.fit_transform(Sentences)
print(TermDocumentMatrix)
```

    [[0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1
      1 1 0 1 0 0 1 0]
     [0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0
      0 1 0 0 0 0 1 0]
     [0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0
      0 1 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 1
      0 0 0 1 1 0 1 0]
     [1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0
      0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 1 0 0 1 0 0 0
      0 0 0 0 0 1 0 0]
     [0 1 0 0 0 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0
      0 0 0 0 0 1 0 1]
     [0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
      1 0 0 0 0 0 0 0]]



```python
#Print the vocabulary: https://en.wikipedia.org/wiki/Vocabulary
print(mlb.classes_)
```

    [',' ':' 'A' 'ABC' 'EPS' 'Graph' 'Human' 'IV' 'Relation' 'System' 'The'
     'Widths' 'and' 'applications' 'binary' 'computer' 'engineering' 'error'
     'for' 'generation' 'graph' 'human' 'in' 'interface' 'intersection'
     'machine' 'management' 'measurement' 'minors' 'of' 'opinion' 'ordered'
     'paths' 'perceived' 'random' 'response' 'survey' 'system' 'testing'
     'time' 'to' 'trees' 'user' 'well-quasi-ordering']



```python
U, s, VT = svd(TermDocumentMatrix)
print(s)
```

    [4.20036444 3.26507534 2.93896964 2.80284675 2.67501894 2.30489547
     2.11326817 2.01596461 1.48473703]



```python
#Select only first 2 Eigen values
s = s[0:2]
print(s)
```

    [4.20036444 3.26507534]



```python
#Now, let's start reducing the dimension of the U and V matrices
#Remember, in matrix multiplication, the first matrix's number of columns 
#must be equal to the number of rows in the second matrix.

#Print the matrix dimensions
print(U.shape)
print(VT.shape)
print(s.shape)

U_small = U[:,0:2]
VT_small = VT[:,0:2]
print(U_small)
print(VT_small)

ReducedU_small = U_small * s
ReducedVT_small = VT_small * s
print(ReducedU_small)
print(ReducedVT_small)
```

    (9, 9)
    (44, 44)
    (2,)
    [[-0.0695516   0.16994215]
     [-0.49804006  0.45363237]
     [-0.24220186  0.16847776]
     [-0.34568692 -0.02716658]
     [-0.41874979  0.44302568]
     [-0.33436179 -0.3679576 ]
     [-0.30985397 -0.31271465]
     [-0.39501539 -0.53942813]
     [-0.17251568 -0.12560594]]
    [[-0.07960304 -0.13511472]
     [-0.11269498 -0.20368108]
     [-0.16577542  0.34144645]
     [-0.07336996  0.01343862]
     [-0.06393409 -0.07855938]
     [-0.07367755 -0.02966816]
     [-0.30239627  0.01105237]
     [-0.00896461 -0.14451447]
     [ 0.05369061  0.2053355 ]
     [ 0.10464536 -0.0045187 ]
     [-0.06470934 -0.07168982]
     [ 0.02316516 -0.05441941]
     [ 0.12781052 -0.05893811]
     [-0.29519252  0.01563709]
     [-0.195909    0.08184599]
     [-0.30829148 -0.15390556]
     [ 0.10464536 -0.0045187 ]
     [ 0.01694401 -0.08004795]
     [-0.29519252  0.01563709]
     [-0.195909    0.08184599]
     [ 0.17917785  0.00292529]
     [ 0.10464536 -0.0045187 ]
     [ 0.17917785  0.00292529]
     [-0.34317071 -0.14082401]
     [ 0.17917785  0.00292529]
     [-0.29519252  0.01563709]
     [-0.04797819 -0.1564611 ]
     [ 0.01694401 -0.08004795]
     [-0.00644055 -0.19971938]
     [ 0.11492442 -0.22375743]
     [-0.01309895 -0.16954265]
     [-0.195909    0.08184599]
     [ 0.17917785  0.00292529]
     [ 0.01694401 -0.08004795]
     [-0.195909    0.08184599]
     [ 0.00384506 -0.2495906 ]
     [-0.04270466 -0.31484262]
     [ 0.04356822 -0.33052245]
     [ 0.10464536 -0.0045187 ]
     [ 0.00384506 -0.2495906 ]
     [ 0.01694401 -0.08004795]
     [ 0.00643401  0.03035187]
     [-0.04413313 -0.40605169]
     [ 0.02316516 -0.05441941]]
    [[-0.29214205  0.55487394]
     [-2.09194975  1.48114385]
     [-1.01733607  0.55009257]
     [-1.45201103 -0.08870094]
     [-1.75890173  1.44651222]
     [-1.40444138 -1.20140927]
     [-1.30149958 -1.02103691]
     [-1.65920858 -1.76127349]
     [-0.72462871 -0.41011285]]
    [[-0.33436179 -0.44115974]
     [-0.47335998 -0.66503407]
     [-0.69631717  1.11484838]
     [-0.30818058  0.04387809]
     [-0.26854648 -0.25650228]
     [-0.30947255 -0.09686876]
     [-1.27017453  0.03608681]
     [-0.03765462 -0.47185063]
     [ 0.22552011  0.67043586]
     [ 0.43954864 -0.01475391]
     [-0.2718028  -0.23407266]
     [ 0.09730212 -0.17768347]
     [ 0.53685076 -0.19243738]
     [-1.23991618  0.05105627]
     [-0.82288921  0.26723331]
     [-1.29493655 -0.50251326]
     [ 0.43954864 -0.01475391]
     [ 0.071171   -0.26136257]
     [-1.23991618  0.05105627]
     [-0.82288921  0.26723331]
     [ 0.75261228  0.00955129]
     [ 0.43954864 -0.01475391]
     [ 0.75261228  0.00955129]
     [-1.44144206 -0.45980099]
     [ 0.75261228  0.00955129]
     [-1.23991618  0.05105627]
     [-0.20152587 -0.51085726]
     [ 0.071171   -0.26136257]
     [-0.02705267 -0.65209881]
     [ 0.48272446 -0.73058488]
     [-0.05502037 -0.55356953]
     [-0.82288921  0.26723331]
     [ 0.75261228  0.00955129]
     [ 0.071171   -0.26136257]
     [-0.82288921  0.26723331]
     [ 0.01615063 -0.8149321 ]
     [-0.17937515 -1.02798487]
     [ 0.1830024  -1.0791807 ]
     [ 0.43954864 -0.01475391]
     [ 0.01615063 -0.8149321 ]
     [ 0.071171   -0.26136257]
     [ 0.02702519  0.09910113]
     [-0.18537524 -1.32578936]
     [ 0.09730212 -0.17768347]]



```python
#Plot matrices
scatterplotmatrix(ReducedVT_small, figsize=(10, 8), names = mlb.classes_)
plt.tight_layout()
plt.show()
```


    
![png](img/output_8_0_0.png)
    



```python
fig, ax = plt.subplots()
x = ReducedVT_small[:,0]
y = ReducedVT_small[:,1]
n = mlb.classes_
ax.scatter(y,x)
for i, txt in enumerate(n):
    ax.annotate(txt, (y[i], x[i]))
```


    
![png](img/output_9_0.png)
    



```python

```
