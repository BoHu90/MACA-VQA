```markdown
MACA-VQA: Quality Assessment of UGC Videos via Multi-level Distortion Adaptation and Spatiotemporal Cross-Attention Fusion
```

### Install Requirements

```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

### Train models

#### 1. Extract video frames

```
python extract_frame/extract_frame_*.py
```

#### 2. Extract motion features

```markdown
use [VideoMae V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py) extract motion features
```

#### 3. Extract distortion features

```markdown
use [Re-IQA](https://github.com/avinabsaha/ReIQA/blob/main/demo_quality_aware_feats.py) extract distortion features
```

#### 4. Train the model

##### 4.1 train on LSVQ

```shell
python -u train_baseline_modular.py --database LSVQ
```

##### 4.2 fine-tune on other dataset

```shell
python -u train_other_modular.py
```

##### 4.3 test on other dataset

```shell
python -u test_baseline_modular.py
```

### Acknowledgement
The basic code is partially from the below repos.
- [ModularVQA]([https://github.com/sunwei925/SimpleVQA](https://github.com/winwinwenwen77/ModularBVQA))
- [Re-IQA]([https://github.com/VQAssessment/DOVER](https://github.com/avinabsaha/ReIQA)
- [VideoMae V2](https://github.com/OpenGVLab/VideoMAEv2)



