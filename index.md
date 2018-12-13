---
title: CS109A Final Project
---

Group #100: Andreas Rousing and Evan DeFilippis

## Dog Breed Classification

### Motivation

>[In Dog Breed Identification: What kind of dog is that?](https://sheltermedicine.vetmed.ufl.edu/library/research-studies/current-studies/dog-breeds/), the authors show that experts identified the prominent
breed correctly only 27% of the time.

Identifying dog breeds correctly has implications not only for policies and dog shelter adoption strategies, but also in breeding contexts. The problem of identifying breeds also affects breed specific legislation (BSL) which often bans a particular breed deemed to be dangerous, from countries ([Denmark has banned 13 different dog breeds and mixed-breeds thereof](https://www.foedevarestyrelsen.dk/english/ImportExport/Travelling_with_pet_animals/Pages/The-Danish-dog-legislation.aspx)), individual cities, neighborhoods and even individual apartment complexes. This has in [some cases](https://www.denverpost.com/2018/01/28/pit-bull-bans-denver-area/) forced tenants out of their homes due to harbouring a banned breed, which in some cases even was been misidentified due to the poor breed classification skills of humans. Much evidence also suggests that BSL is not effective when it comes to reduction in attacks or dog bites [1], [4], [6]. Work in [2] suggests other negative aspects of BSL such as misconceptions of breeds contributing to biased reports of dog attacks, leading to poorer shelter adoption rates.


### Problem Statement
The goal of this project is to devise and evaluate a deep convolutional neural net model, that can distinguish between the 120 pure breeds of dogs with an accuracy above 30% accuracy. We have achieved this by experimenting and testing several networks, architechtures and with a thorough data augmentation pipeline using Keras, ImageDataGenerator, in addition to methods such as batch normalization, and drop-out layers in order to minimize overfitting. This have led to a classification algorithm much better than dog breed experts, as described in [In Dog Breed Identification: What kind of dog is that?](https://sheltermedicine.vetmed.ufl.edu/library/research-studies/current-studies/dog-breeds/).

**High-level project goals**
1. Build a few models to classify dog breeds.
2. Evaluate the predictive quality of the models.
3. Compare the results from each model.
4. Discuss the relative merits of each model.
5. Devise a model with a higher than 30% accuracy on test set*
6. Use the model to determine outliers.

Since we were able to reach a goal of above 30% accuracy we did not have to limit the number of classes we are working with or combine them into superclasses in order to simplify the classification problem.

*Group set goal - achieved