---
nav_include: 5
title: Future Work
---

## Contents
{:.no_toc}
*  
{: toc}

## 22. Future Work

In the future, we'd like to explore a huge amount of topics and ideas for improvement we stumbled upon in researching for this project.  


*   As mentioned above, we had a concern that there may be something problematic about cropping both the training and test set images, even though the cropping boundaries were provided in the original dataset.  One potential idea is to train yet another classifier that is given bounding box information of a given image, and train a neural network to estimate a bounding box on a left-out test set.  You would then use these two classifiers in concert with one another -- one classifier would predict a bounding box around the dog and crop the image, and the other classifier would predict on the cropped image.  

*  We wanted to experiment with GANs and other non-CNN networks in order to get our feet wet.  Unfortunately, we couldn't find an application that was actuall suited for the prediction task at hand.  One potential application is the use of GANs to enhance image augmentation, which has been done as recently as [2017](https://arxiv.org/pdf/1801.06665.pdf) on the Stanford Dog dataset

* We would like to, in the future, learn how to generate adverserial examples to force our classifier to misclassify a given image with the addition of some noise.  One interesting goal would be to craft a dog classifier that is robust to adverserial images. 

* It would be interesting to experiment with non CNN techniques, such as SVM's with linear kernels.  We have seen [papers ](https://web.stanford.edu/class/cs231a/prev_projects_2016/output%20[1].pdf)that utilize interesting techniques such as 'facial keypoint detection' to extract key characteristics about dog faces using a CNN before doing SVM analysis. 

