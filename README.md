# 🤖⚡ Cross-Device Collaborative Test-Time Adaptation

<p align="center">
<img src="figures/setting.png" alt="settings" width="100%" align=center />
</p>

This is the official project repository for [Cross-Device Collaborative Test-Time Adaptation (NeurIPS 2024)](https://openreview.net/pdf?id=YyMiO0DWmI) by Guohao Chen, Shuaicheng Niu, Deyu Chen, Shuhai Zhang, Changsheng Li, Yuanqing Li and Mingkui Tan

* 1️⃣ CoLA conducts TTA in a multi-device collaborative manner. It enables knowledge accumulation, sharing, and exploitation across devices and heterogeneous scenearios to boost adaptation efficiency and performance, while keeping privacy preserved and communication efficient.


* 2️⃣ CoLA devise two collaboration strategies (_BP-Based_ & _Forward-Only_) to address a practical scenario where multiple devices with different computational resources and latency requirements need to perform TTA simultaneously. Our CoLA paradigm is decentralized and flexible, which allows all agents to join or leave the collaboration at any time.

<p align="center">
<img src="figures/method.png" alt="settings" width="90%" align=center />
</p>


**Dependencies Installation:**
```
conda create -n cola python=3.8.13
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install timm==0.6.12
```

**Data Preparation:**

This repository contains code for evaluation on ImageNet-C/R/A/V2/Sketch with Vit-Base. But feel free to use your own data and models! Please check [here 🔗](dataset/README.md) for a detailed guide on preparing these datasets.

# Example: ImageNet-C Experiments

For lifelong test-time adaptation experiments, simply run the following commands
```
python3 main_lifelong.py \
    --data path/to/imagenet \
    --data_v2 path/to/imagenet-v2 \
    --data_sketch path/to/imagenet-sketch \
    --data_corruption path/to/imagenet-c \
    --data_rendition path/to/imagenet-r \
    --resume weights/original.pth \
    --algorithm [tent/cotta/eata/t3a/lame/eta(-cola)/sar(-cola)/deyo(-cola)]
```
included in ``main_lifelong.sh``. Please refer to ``main_bp_collboration.sh, main_fp_agents.sh, main_single_domain.sh`` for collobrative and single-domain TTA experiments, respectively. We serve as a plug-and-play module to enhance TTA performance across scenarios.
* on lifelong TTA: SAR (60.5%) _vs._ SAR+CoLA (64.0%), and ETA (46.4%) _vs._ ETA+CoLA (64.8%), see Table 1.
* on bp collaborative TTA: achieves an up to 78.0 times speed up in sample efficiency on ETA, see Figure 3.
* on single-domain TTA over mild and wild: SAR (56.1%) _vs._ SAR+CoLA (58.1%), and ETA (56.1%) _vs._ ETA+CoLA (59.3%), see Table 4.
<!-- * on fp collaborative TTA: improves accuracy by over 30% on ImAgeNet-C with efficiency similar to standard inference, see Table 3 and 5. -->


# Correspondence

Please contact Guohao Chen by [chenguohao987 at gmail.com] and Shuaicheng Niu by [shuaicheng.niu at ntu.edu.sg] if you have any questions. 📬


# Citation

If you find our plug-and-play CoLA method for lifelong, collaborative, single-domain TTA enhancement—or our cross-device collaborative TTA setting—beneficial to your research, please consider citing our paper:

```
@inproceedings{chen2024cross,
  title={Cross-Device Collaborative Test-Time Adaptation},
  author={Chen, Guohao and Niu, Shuaicheng and Chen, Deyu and Zhang, Shuhai and Li, Changsheng and Li, Yuanqing and Tan, Mingkui},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

# Acknowledgement
The code is inspired by [EATA 🔗](https://github.com/mr-eggplant/EATA) and [FOA 🔗](https://github.com/mr-eggplant/FOA)