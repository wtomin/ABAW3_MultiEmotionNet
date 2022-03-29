# The Multitask EmotionNet in the ABAW3 challenge

We participated in the [ABAW3 MTL Challenge](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) held in conjunction with CVPR2022. Our team name is NISL-2022. 

Leaderboard is [here](https://drive.google.com/file/d/1rTLFTQaVZOrtB17WrTOH7m038TwdLIPB/view). We won the first place in the MTL challenge!

This repository contains the code for our Multitask EmotionNet (the static and temporal approaches).

# Dependency

Install the dependencies with the `requirements.txt` in `MTL/`:
```
pip install requirements.txt
```

# Model Architecture

![image info](./MTL/figures/Model_Architecture.jpg)


# Pretrained Model

The pretrained model can be downloaded from this [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ddeng_connect_ust_hk/EaRbMo6Q5uZGtHqQHfGzxigB3xJwzjkGF3qS8hlbw4gVPA?e=0N0Iv3). 

This model was trained on the training set of the Aff-wild2 dataset, and evaluated on the validation set of the Aff-wild2 dataset.

The validation metrics are listed below:

| F1-AU | F1-EXPR | CCC-V | CCC-A|
| --- | ---| ---| ---|
| 0.548| 0.518| 0.447| 0.499|


For more details about this approach, please refer to our [Axiv paper](https://arxiv.org/abs/2203.12845).




