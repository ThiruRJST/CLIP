# Finetuning CLIP on Custom Dataset
- [x] Provide Basic Training Script
- [ ] Provide Captioning script using BLIPv1, BLIPv2, etc;
- [ ] Add metric logging system to the training script
 
 **NOTE: Issues you might encounter**
1. The CLIP model takes batch size as number of classes, so make sure you have **Balanced Batch Sampler** if your number of classes are less than or equal to batch size.
2. Make sure your dataset is clean and there are no leakages of one class data onto the other, This will hinder the model's training progress.
 
