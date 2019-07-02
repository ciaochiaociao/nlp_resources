# NLP Resources
Resources about NLP (Natural Language Processing), especially IE (Information Extraction) and NER (Named-Entity Recognition) 

## STANFORD COURSES

1. CS224N - NLP with Deep Learning: [video link](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) | [course URL](http://web.stanford.edu/class/cs224n/)
(advanced, more related to DNN) 

  All of us in the lab have watched this course. Unlike the following two courses, this course is the most helpful one because it introduces most of the deep neural networks we are going to use in our NLP research. However, this course is heavy and advanced. And it takes much time to understand all of them.

  So I would suggest you watch lessons in this order (you can skip some lessons if you already know) 

  must know: 
  * 1-4: for introduction to Neural Network(NN)
  * 5, 6, 16: for basic (more upstream) NLP tasks with deep learning
  * 6, 7, 11: for useful neural networks RNN-ish models and CNN

  highly recommend:
  * 8, 13, 14: for emerging and really powerful NN model Attention, Self-Attention, Transformer, and BERT

  good to know:
  * 17, 18, 12: for knowing more "information" that you can add to your model
  * 10, 15: for more downstream NLP tasks

2. From Language to Information: [video link](https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA)
(an introductory course with traditional statistical approaches)
The recommended chapter which is related to our research: Ch7 Introduction to Information Retrieval  

3. Natural Language Processing: [video link](https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm)
(an introductory course with traditional statistical approaches)
Lessons highly related to our search 

* [Information Extraction and NER](https://www.youtube.com/watch?v=ARsDDLffoMk&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=45&t=0s) (recommended)
* [NER Evaluation](https://www.youtube.com/watch?v=0qWDkRdWbSw&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=46&t=0s)
* [NER Simplest Model](https://www.youtube.com/watch?v=m8RrR5GORLg&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=47&t=0s) (For DNN model, I suggest to read papers below after you finish the must-know part of the Stanford lessons)

## Where to find papers with codes

the leaderboard of different tasks, the state-of-the-art NER papers and their codes (most of them are open-sourced) 

* [NLP-progress](http://nlpprogress.com/english/named_entity_recognition.html)
* [PapersWithCode](https://paperswithcode.com/task/named-entity-recognition-ner)

Some NER papers from the followed two websites: 

* [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370)
* [Pooled Contextualized Embeddings for Named Entity Recognition](http://alanakbik.github.io/papers/naacl2019_embeddings.pdf)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
* [SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://paperswithcode.com/paper/scibert-pretrained-contextualized-embeddings)
* [Using Similarity Measures to Select Pretraining Data for NER](https://paperswithcode.com/paper/using-similarity-measures-to-select)
* [Extracting Entities and Relations with Joint Minimum Risk Training](https://www.aclweb.org/anthology/D18-1249)

## More about BERT, Transformer, Contextual Embeddings
From Prof. Hung-Yi Lee from National Taiwan University 
* [Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)
* [ELMO, BERT, GPT](https://www.youtube.com/watch?v=UYPa347-DdE)

Articles 
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [Deconstructing BERT Part1](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)
* [Deconstructing BERT Part2](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

Papers 
* [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950)
* [GPT-2](https://towardsdatascience.com/openai-gpt-2-understanding-language-generation-through-visualization-8252f683b2f8?source=user_profile---------0-----------------------)

## Bloggers, Websites, and Podcasts
* [Sebastian Ruders](http://ruder.io/)
* [Towards Data Science](https://towardsdatascience.com)
* [SAIL](http://ai.stanford.edu/blog/)
* [The Gradient](https://thegradient.pub/)
* [Allen AI Podcasts](https://allenai.org/podcasts/podcasts-all.html)

## BOOKS

[A Course in Machine Learning (Hal Daumé III)](http://ciml.info/): If you never took a course of ML, you can go check this out. 

[Speech and Language Processing (3rd ed. draft) (Dan Jurafsky and James H. Martin)](https://web.stanford.edu/~jurafsky/slp3/): Mostly NLP, not much about DNN 

[A Primer on Neural Network Models for Natural Language Processing (Yoav Goldberg)](http://u.cs.biu.ac.il/~yogo/nnlp.pdf): Highly recommended. Both DNN and NLP. 

[Deep Learning (Ian Goodfellow, Yoshua Bengio, and Aaron Courville)](http://www.deeplearningbook.org/): Bible of DNN, but not much about NLP 



Provided by: Chiao-Wei Hsu (Feel free to mail me any questions) 

email: cwhsu@iis.sinica.edu.tw
