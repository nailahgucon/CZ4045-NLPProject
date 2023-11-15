This repository contains the code for SC4002 Group Project.


# Part 1

## Instructions to Run the Code

1. **Requirements**:
* pandas=2.1.1
* python=3.9.18
* numpy=1.26.0
* gensim=4.3.2
* pytorch=2.0.1
* nltk=3.8.1
* scikit-learn=1.3.2
* matplotlib=3.8.1

2. **Setup**:
Install required packages:
!pip install --upgrade gensim 


3. **Running the Code**:
Select “Run all” to run all codes in Google Colab (or Jupyter Notebook)

3. **Input**:
eng.testa, eng.testb, eng.train

#### Output:
Question 1.1
Most similar word to 'student': students (Cosine Similarity: 0.73)
Most similar word to 'Apple': Apple_AAPL (Cosine Similarity: 0.75)
Most similar word to 'apple': apples (Cosine Similarity: 0.72)

Explanation: output showing most similar words and the cosine similarity of these words: ‘student’, ‘Apple’, ‘apple’.

Question 1.2
Total unique words (203621 in total): 17493
Total unique characters: 75
Total unique named entity tags: 19 

Number of sentences in train set: 14041
Number of sentences in development set: 3250
Number of sentences in test set: 3453

Explanation: output showing the number of sentences in train, development and test sets.

Word Labels       #
0        S-ORG    3836
1            O  169578
2       S-MISC    2580
3        B-PER    4284
4        E-PER    4284
5        S-LOC    6099
6        B-ORG    2485
7        E-ORG    2485
8        I-PER     244
9        S-PER    2316
10      B-MISC     858
11      I-MISC     297
12      E-MISC     858
13       I-ORG    1219
14       B-LOC    1041
15       E-LOC    1041
16       I-LOC     116
17     <START>      -1
18      <STOP>      -2

Explanation: printed the complete set of all possible word labels based on the BIOES tagging scheme


Data with BIO tagging scheme VS Data with BIOES tagging scheme

|            | Word                    | BIO           | Word                    | BIOES         |
|----        |-----------------        |-------        |-----------------        |-------        |
| 0          | He                      | O             | He                      | O             |
| 1          | has                     | O             | has                     | O             |
| 2          | spent                   | O             | spent                   | O             |
| 3          | his                     | O             | his                     | O             |
| 4          | adult                   | O             | adult                   | O             |
| 5          | life                    | O             | life                    | O             |
| 6          | in                      | O             | in                      | O             |
| 7          | politics                | O             | politics                | O             |
| 8          | ,                       | O             | ,                       | O             |
| 9          | coming                  | O             | coming                  | O             |
| 10         | to                      | O             | to                      | O             |
| 11         | Congress                | I-ORG         | Congress                | S-ORG         |
| 12         | as                      | O             | as                      | O             |
| 13         | an                      | O             | an                      | O             |
| 14         | aide                    | O             | aide                    | O             |
| 15         | in                      | O             | in                      | O             |
| 16         | 0000                    | O             | 0000                    | O             |
| 17         | after                   | O             | after                   | O             |
| 18         | three                   | O             | three                   | O             |
| 19         | years                   | O             | years                   | O             |
| 20         | in                      | O             | in                      | O             |
| 21         | the                     | O             | the                     | O             |
| 22         | Air                     | I-ORG         | Air                     | B-ORG         |
| 23         | Force                   | I-ORG         | Force                   | E-ORG         |
| 24         | and                     | O             | and                     | O             |
| 25         | then                    | O             | then                    | O             |
| 26         | being                   | O             | being                   | O             |
| 27         | elected                 | O             | elected                 | O             |
| 28         | to                      | O             | to                      | O             |
| 29         | the                     | O             | the                     | O             |
| 30         | House                   | I-ORG         | House                   | B-ORG         |
| 31         | of                      | I-ORG         | of                      | I-ORG         |
| 32         | Representatives         | I-ORG         | Representatives         | E-ORG         |
| 33         | himself                 | O             | himself                 | O             |
| 34         | in                      | O             | in                      | O             |
| 35         | 0000                    | O             | 0000                    | O             |
| 36         | .                       | O             | .                       | O             


#### Explanation:
We made use of the BIOES tagging scheme as it gives a more accurate representation when differentiating multi-word and single-word entities as compared to BIO. With the End tag “E”, BIOES tagging scheme could help with identifying the last word of a name entity. The BIOES tagging scheme is as shown in TABLE 4.

Air Force: B-ORG is assigned to “Air” as it represents the beginning of an organisation phrase while E-ORG is assigned to “Force” as it represents the end of the organisation phrase, therefore, a complete named entity is formed.


House of Representatives: B-ORG is assigned to “House” as it represents the beginning of an organisation phrase, I-ORG is assigned to and E-ORG is assigned to “Force” as it represents the end of the organisation phrase, therefore, a complete named entity is formed.

Question 1.3
-------------- Epoch 1 --------------
2000 :  0.5522299778563758
4000 :  0.25782368293091373
6000 :  0.19516489536330967
8000 :  0.16203415877294766
10000 :  0.1493797413448389
12000 :  0.1293671415964332
14000 :  0.11344206438027148
Train: new_F1_Score: 0.0038588754134509366 best_F1_score: -1.0 
Dev: new_F1_Score: 0.004708375022070507 best_F1_score: 0.0 
Test: new_F1_Score: 0.0032389635316698658 best_F1_score: -1.0 
F1 Score of Test Set(Epoch1): 0.0032389635316698658
F1 Score of Development Set(Epoch1): 0.004708375022070507
Saving Model
The running time to run each epoch: 3289.4680638313293 s
-------------- Epoch 2 --------------
16000 :  0.08872051436250757
18000 :  0.07983936756161944
20000 :  0.07713142587643271
22000 :  0.07031296797524343
24000 :  0.06499001109209744
26000 :  0.07258709765849695
28000 :  0.05208072451239837
Train: new_F1_Score: 0.0031744148099801224 best_F1_score: 0.0038588754134509366 
Dev: new_F1_Score: 0.0035279805352798053 best_F1_score: 0.004708375022070507 
Test: new_F1_Score: 0.002115480338476854 best_F1_score: 0.0032389635316698658 
F1 Score of Test Set(Epoch2): 0.002115480338476854
F1 Score of Development Set(Epoch2): 0.0035279805352798053
The running time to run each epoch: 2918.7407414913177 s
-------------- Epoch 3 --------------
30000 :  0.04271275091618329
32000 :  0.03464456954959714
34000 :  0.05290799626705991
36000 :  0.03844789970907775
38000 :  0.04399509331182097
40000 :  0.04455220374872002
42000 :  0.033208939886481194
Train: new_F1_Score: 0.0026710499280871167 best_F1_score: 0.0038588754134509366 
Dev: new_F1_Score: 0.004057279236276849 best_F1_score: 0.004708375022070507 
Test: new_F1_Score: 0.0025713236194441044 best_F1_score: 0.0032389635316698658 
F1 Score of Test Set(Epoch3): 0.0025713236194441044
F1 Score of Development Set(Epoch3): 0.004057279236276849
The running time to run each epoch: 2921.7061953544617 s
-------------- Epoch 4 --------------
44000 :  0.029378174914300323
46000 :  0.028620454751100827
48000 :  0.025971244044986325
50000 :  0.03316411869314203
52000 :  0.02684147233620784
54000 :  0.02631890524123787
56000 :  0.026534681770661202
Train: new_F1_Score: 0.002326325278432057 best_F1_score: 0.0038588754134509366 
Dev: new_F1_Score: 0.003419206508282733 best_F1_score: 0.004708375022070507 
Test: new_F1_Score: 0.0024148756339048543 best_F1_score: 0.0032389635316698658 
F1 Score of Test Set(Epoch4): 0.0024148756339048543
F1 Score of Development Set(Epoch4): 0.003419206508282733
The running time to run each epoch: 2917.4237031936646 s
-------------- Epoch 5 --------------
2000 :  0.07086782180622597
4000 :  0.08554724193884479
6000 :  0.0882462767384268
8000 :  0.07053660018359598
10000 :  0.07727696661831275
12000 :  0.060795362047532456
14000 :  0.05482337219255637
Train: new_F1_Score: 0.003120534282131296 best_F1_score: -1.0 
Dev: new_F1_Score: 0.004718372161604247 best_F1_score: 0.0 
Test: new_F1_Score: 0.003620564808110065 best_F1_score: -1.0 
F1 Score of Test Set(Epoch5): 0.003620564808110065
F1 Score of Development Set(Epoch5): 0.004718372161604247
Saving Model
The running time to run each epoch: 3036.317501783371 s
-------------- Epoch 6 --------------
2000 :  0.02285720047948826
4000 :  0.030650647315032014
6000 :  0.021808663510211913
8000 :  0.034182661143227794
10000 :  0.02546489253524515
12000 :  0.02881385992813444
14000 :  0.030312976571662237
Train: new_F1_Score: 0.00233730187714557 best_F1_score: -1.0 
Dev: new_F1_Score: 0.004267931238885595 best_F1_score: 0.0 
Test: new_F1_Score: 0.002432350258437215 best_F1_score: -1.0 
F1 Score of Test Set(Epoch6): 0.002432350258437215
F1 Score of Development Set(Epoch6): 0.004267931238885595
Saving Model
The running time to run each epoch: 3154.043802022934 s
-------------- Epoch 7 --------------
16000 :  0.019515358599228563
18000 :  0.02514976546555685
20000 :  0.020028070514093003
22000 :  0.021759246045981187
24000 :  0.01970245459350357
26000 :  0.020783244984332285
28000 :  0.014455211009082916
Train: new_F1_Score: 0.0021224940759155078 best_F1_score: 0.00233730187714557 
Dev: new_F1_Score: 0.00319337670017741 best_F1_score: 0.004267931238885595 
Test: new_F1_Score: 0.0023010778733196075 best_F1_score: 0.002432350258437215 
F1 Score of Test Set(Epoch7): 0.0023010778733196075
F1 Score of Development Set(Epoch7): 0.00319337670017741
The running time to run each epoch: 2912.550426721573 s
-------------- Epoch 8 --------------
30000 :  0.016100294631066044
32000 :  0.014553410543503984
34000 :  0.01424637520386563
36000 :  0.01613811069999507
38000 :  0.010956559274768471
40000 :  0.019878571059055865
42000 :  0.019007541300062505
Train: new_F1_Score: 0.0023419880876655295 best_F1_score: 0.00233730187714557 
Dev: new_F1_Score: 0.003376804843968328 best_F1_score: 0.004267931238885595 
Test: new_F1_Score: 0.0028469750889679713 best_F1_score: 0.002432350258437215 
F1 Score of Test Set(Epoch8): 0.0028469750889679713
F1 Score of Development Set(Epoch8): 0.003376804843968328
The running time to run each epoch: 2909.1956672668457 s
-------------- Epoch 9 --------------
44000 :  0.013241294037032468
46000 :  0.009754542931285706
48000 :  0.010686486612412626
50000 :  0.016412198414456555
52000 :  0.013450998163356355
54000 :  0.01717139370648988
56000 :  0.013170531287194255
Train: new_F1_Score: 0.002235707441711913 best_F1_score: 0.0023419880876655295 
Dev: new_F1_Score: 0.003173110823833588 best_F1_score: 0.004267931238885595 
Test: new_F1_Score: 0.002287778446718844 best_F1_score: 0.0028469750889679713 
F1 Score of Test Set(Epoch9): 0.002287778446718844
F1 Score of Development Set(Epoch9): 0.003173110823833588
The running time to run each epoch: 2908.2078976631165 s
The running time to perform training: 11883.999492645264 s
-------------- Epoch 10 --------------
2000 :  0.017801072055355385
4000 :  0.02309376797811614
6000 :  0.016408120106657473
8000 :  0.024127455268095822
10000 :  0.01920241689763682
12000 :  0.022950091924196146
14000 :  0.022940817097537113
Train: new_F1_Score: 0.002418062636562273 best_F1_score: -1.0 
Dev: new_F1_Score: 0.003929507025482257 best_F1_score: 0.0 
Test: new_F1_Score: 0.002190847127555988 best_F1_score: -1.0 
F1 Score of Test Set(Epoch10): 0.002190847127555988
F1 Score of Development Set(Epoch10): 0.003929507025482257
Saving Model
The running time to run each epoch: 2965.505757331848 s
The running time to perform training: 2965.5204224586487 s

Total running time:
29933.176120758057s

### Explanation:
Under each epoch, the loss function for every 2000 iterations is printed out along with the F1_score for Training dataset, development dataset and testing dataset. Some values for best_F1_score are -1.0, 0.0 and -1.0 for Training dataset, development dataset and testing dataset which is the initialisation of the best F1-scores. Once the score for F1-score is higher, the best F1_score is replaced with the current high F1-score. The time taken to run each epoch is printed out.  The total running time for the overall run of all the epochs is printed at the end which is 29933.176120758057 s. There may be some inconsistencies in the F1-scores and running time due to the reload functionality that we implemented. Due to google colab gpu limitations, each run can only execute up to a few epochs. The model is required to be reloaded multiple times to complete 10 epochs.

#### Link to Trained Model: https://tinyurl.com/CZ4042-G38-model 

### Part 2

Requirements:
* gensim==4.3.2
* matplotlib==3.7.2
* nltk==3.8.1
* numpy==1.25.1
* pandas==1.4.1
* scikit_learn==1.3.0
* torch==2.0.1

#### Input:
TREC_train.csv, TREC_test.csv

### Output:
Fold 1
Training and validating model
Early stopping
validation accuracy = 59.70%
Fold 2
Training and validating model
Early stopping
validation accuracy = 34.85%
Fold 3
Training and validating model
Early stopping
validation accuracy = 83.92%
Fold 4
Training and validating model
Early stopping
validation accuracy = 86.25%
Fold 5
Training and validating model
Early stopping
validation accuracy = 76.67%
Mean cross-validation accuracy = 68.28%

### Explanation:
Each fold has a validation accuracy for each subsets of data. A mean is calculated over all folds, representing a generalized accuracy.

### Output:
Training and validating model
------------------------- Epoch 1 -------------------------
Training accuracy: 33.99%
Training loss: 1.5395
Validation accuracy: 42.40%
Validation loss: 1.5111


------------------------- Epoch 2 -------------------------
Training accuracy: 46.36%
Training loss: 1.4419
Validation accuracy: 51.00%
Validation loss: 1.3888


------------------------- Epoch 3 -------------------------
Training accuracy: 59.09%
Training loss: 1.3209
Validation accuracy: 67.20%
Validation loss: 1.2507


------------------------- Epoch 4 -------------------------
Training accuracy: 69.86%
Training loss: 1.2220
Validation accuracy: 69.00%
Validation loss: 1.2096


------------------------- Epoch 5 -------------------------
Training accuracy: 78.96%
Training loss: 1.1340
Validation accuracy: 83.80%
Validation loss: 1.0952


------------------------- Epoch 6 -------------------------
Training accuracy: 89.49%
Training loss: 1.0235
Validation accuracy: 86.40%
Validation loss: 1.0398


------------------------- Epoch 7 -------------------------
Training accuracy: 91.54%
Training loss: 0.9996
Validation accuracy: 85.00%
Validation loss: 1.0578


------------------------- Epoch 8 -------------------------
Training accuracy: 94.08%
Training loss: 0.9708
Validation accuracy: 86.80%
Validation loss: 1.0357


------------------------- Epoch 9 -------------------------
Training accuracy: 94.90%
Training loss: 0.9617
Validation accuracy: 88.40%
Validation loss: 1.0228


------------------------- Epoch 10 -------------------------
Training accuracy: 95.51%
Training loss: 0.9544
Validation accuracy: 88.80%
Validation loss: 1.0170


------------------------- Epoch 11 -------------------------
Training accuracy: 96.39%
Training loss: 0.9445
Validation accuracy: 90.40%
Validation loss: 1.0050


------------------------- Epoch 12 -------------------------
Training accuracy: 96.68%
Training loss: 0.9409
Validation accuracy: 90.20%
Validation loss: 1.0080


------------------------- Epoch 13 -------------------------
Training accuracy: 97.13%
Training loss: 0.9352
Validation accuracy: 90.40%
Validation loss: 1.0042


------------------------- Epoch 14 -------------------------
Training accuracy: 97.48%
Training loss: 0.9319
Validation accuracy: 90.40%
Validation loss: 1.0071


------------------------- Epoch 15 -------------------------
Training accuracy: 97.52%
Training loss: 0.9301
Validation accuracy: 90.00%
Validation loss: 1.0070


------------------------- Epoch 16 -------------------------
Training accuracy: 97.62%
Training loss: 0.9288
Validation accuracy: 91.20%
Validation loss: 1.0024


------------------------- Epoch 17 -------------------------
Training accuracy: 97.73%
Training loss: 0.9276
Validation accuracy: 89.80%
Validation loss: 1.0064


------------------------- Epoch 18 -------------------------
Training accuracy: 97.81%
Training loss: 0.9268
Validation accuracy: 90.20%
Validation loss: 1.0057


------------------------- Epoch 19 -------------------------
Training accuracy: 97.85%
Training loss: 0.9275
Validation accuracy: 90.20%
Validation loss: 1.0062

Early stopping
Time used for training is: 38.652079582214355

#### Explanation:
Training accuracy, Training Loss, validation accuracy and validation loss is printed for each epoch of training. The running time is printed at the end.

#### Output:
Testing Model
Test loss: 0.9842
Test accuracy: 0.9220

#### Explanation:
The output shows the testing of the Bidirectional LSTM model. The testing returns an accuracy of 92.20% with the loss of 0.9842.