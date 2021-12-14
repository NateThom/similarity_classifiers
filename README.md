## Unlabeled Facial Verification and Doppelganger Detection
This repository is currently dedicated to exploring a new research direction. The goal of this work is to learn representations of faces that "make sense". Here, making sense means that the learned representation for a human face should align with expectations that humans might have for the face. 

For example, facial images of siblings or doppelgangers should appear close to each other in latent space. Another example might be representation arithmetic. The representation vector for a father is subtracted from the respresentation vector of the son. This should result in a feature vector which is close to individuals who look like the mother and son, but not the father.

I hope to achieve this goal with unsupervised loss functions such as contrastive loss (presented in SimCLR) or triplet loss.

SimCLR implementation based on the work of Janne Spijkervet (https://github.com/Spijkervet, https://github.com/spijkervet/SimCLR)

## Progress

### Using SimCLR

### Using SimCLR as a pretraining step for the task of identity recognition

### Current: techniques for analyzing the learned latent space