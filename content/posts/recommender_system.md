---
title: "Recommender Systems and Applications of Berlin Grocery Shopping"
date: 2023-06-28T13:46:50+08:00
draft: False
# cover:
#     image: /images/result_vbt.png
#     alt: MA crossing backtest from VBT
#     caption: MA crossing backtest from VBT
tags: ['Recommender Systems', 'Cosine Similarity', 'Content-based Filtering']
author: ['Jia-Hau Ching']
---

As amount of data explosively increasing nowadays, we have got so many information to digest. On the one hand it is healthy growth of society for the diversity of information, but on the other hand having too many options distracts our attention and perhaps it is not efficient to obtain the knowledge we need. In order to filter out the most relevant information, recommender systems are adopted. Recommender systems are filtering models for recommended information to people. Generally they are categorized into three types.
1. Content-based filtering
2. Collaborative filtering
3. Hybrid filtering

Generally, the first method recommends items from the input of users. It could be the information users provided when setting up their account like Twitch, Netflix, or just from a search query. Then using algorithms to quantify and compare similarity between user input and database. The second one uses other user records and manage to recommend items that other similar users bought. The third is the combination of these two models for solving the drawbacks of each one. In this post, it will focus on content-based filtering model and will give an example of using it. It is a cooperative project of building a recommender system for Berlin grocery shopping on [Omdena](https://omdena.com/chapter-challenges/developing-a-recommended-system-for-grocery-shopping-in-berlin/). Our objective is to create a tool for people, especially new comers, to help them explore the ideal stores for grocery shopping.

Due to lack of the information of customer profiles, we chose to develop an application based on a content-based filtering model. First, we collect the public data from different grocery stores in Berlin through web scraping. After organizing and cleaning the data, in order to avoid recommending the most frequent items, the cosine similarity is chosen as measurement for product similarity. We build several models and compare their performance by examining the recommended results of each query. One of the applications uses the model,  [Sentence Transformer](https://www.sbert.net/) for analyzing contextual queries. What makes it unique is that it can understand more vague queries than other models. It gives somehow relevant results when queries are something like "gift for Christmas". Give it a try! You can access it in the [**Apps**](/apps/) or by this [link](https://huggingface.co/spaces/jiahau/Rec-sys-Berlin-ST).




<!-- (--- in processing ---)

How to select features

candidate generation, scoring and re-ranking
sequence learning
Neural network model
Embedding space
similarity measures: Cosine, Dot-product, Euclidean Distance
pros and cons of similarity measures
pros and cons of content-based filtering
pros and cons of collaborative filtering
Matrix Factorization
Ways to minimize the objective function
1. Stochastic gradient descend (SGD)
2. Weighted alternating least squares (WALS)
Target encoding
TensorFlow, sparsetensor
Regularization
t-SNE
High norm problem
Folding problem -->