English | [简体中文](README.md)

## INTRODUCTION
PaddleOCR aims to create a rich, leading, and practical OCR tools that help users train better models and apply them into practice.

**Live stream on coming day**:  July 21, 2020 at 8 pm BiliBili station live stream,

**Recent updates**
-2020.7.15, add mobile App demo , support both iOS and  Android  ( based on easyedge and Paddle Lite)
-2020.7.15, improve the  deployment ability, add the C + +  inference , serving deployment. In addtion, the benchmarks of the ultra-lightweight Chinese OCR model are provided.
-2020.7.15, add several related datasets, data annotation and synthesis tools.
- 2020.7.9 Add recognition model to support space, [recognition result](#space Chinese OCR results). For more information: [Recognition](./doc/doc_en/recognition_en.md) and [quickstart](./doc/doc_en/quickstart_en.md)
- 2020.7.9 Add data auguments and learning rate decay strategies,please read [config](./doc/doc_en/config_en.md)
- 2020.6.8 Add [dataset](./doc/doc_en/datasets_en.md) and keep updating
- 2020.6.5 Support exporting `attention` model to `inference_model`
- 2020.6.5 Support separate prediction and recognition, output result score
- [more](./doc/doc_en/update_en.md)

## FEATURES
- Ultra-lightweight Chinese OCR model, total model size is only 8.6M
    - Single model supports Chinese and English numbers combination recognition, vertical text recognition, long text recognition
    - Detection model DB (4.1M) + recognition model CRNN (4.5M)
- Various text detection algorithms: EAST, DB
- Various text recognition algorithms: Rosetta, CRNN, STAR-Net, RARE
- Support Linux, Windows, MacOS and other systems.



## Visualization

![](doc/imgs_results/11.jpg)

[More visualization]((./doc/doc_en/visualization_en.md))



You can also quickly experience the ultra-lightweight Chinese OCR : [Online Experience](https://www.paddlepaddle.org.cn/hub/scene/ocr) **

Mobile DEMO experience (based on EasyEdge and Paddle-Lite, supports iOS and Android systems): [SIgn in the website to obtain the QR code for  installing the App](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)

 Also, you can scan the QR code blow to install the App (**Android support only**)

<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

- [**OCR Quick Start**](./doc/doc_en/quickstart_en.md)

<a name="Supported-Chinese-model-list"></a>

### Supported Chinese models list:

|Model Name|Description |Detection Model link|Recognition Model link| Support for space Recognition Model link|
|-|-|-|-|-|
|chinese_db_crnn_mobile|ultra-lightweight Chinese OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [pre-train model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server|General Chinese OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [pre-train model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)


For testing our Chinese OCR online：https://www.paddlepaddle.org.cn/hub/scene/ocr




## 文档教程
- [Installation](./doc/doc_en/installation_en.md)
- [Quick Start](./doc/doc_en/quickstart_en.md)
- Algorithm introduction
    - [TEXT DETECTION ALGORITHM](#TEXT DETECTION ALGORITHM)
    - [TEXT RECOGNITION ALGORITHM](#TEXT RECOGNITION ALGORITHM)
    - [ END-TO-END OCR ALGORITHM](# END-TO-END OCR ALGORITHM)
- Model training/evaluation
    - [TEXT DETECTION](./doc/doc_en/detection_en.md)
    - [TEXT RECOGNITION](./doc/doc_en/recognition_en.md)
    - [yml configuration](./doc/doc_en/config_en.md)
    - [Important tricks](./doc/doc_en/tricks_en.md)
- 预测部署
    - [Python Inference](./doc/doc_en/inference_en.md)
    - [C++ Inference](./deploy/cpp_infer/readme_en.md)
    - [Serving](./doc/doc_en/serving_en.md)
    - [Moile](./deploy/lite/readme_en.md)
    - Slim（coming soon）
    - [Benchmark](./doc/doc_en/benchmark_en.md)
- 数据集
    - [General Print OCR Datasets(Chinese/English)](./doc/doc_en/datasets_en.md)
    - [HandWritten_OCR_Datasets(Chinese)](./doc/doc_en/handwritten_datasets_en.md)
    - [Various OCR Datasets(multilingual)](./doc/doc_en/vertical_and_multilingual_datasets_en.md)
    - [Data Annotation Tools](./doc/doc_en/data_annotation_en.md)
    - [Data Synthesis Tools](./doc/doc_en/data_synthesis_en.md)
- [FAQ](#FAQ)
- Visualization
    - [Ultra-lightweight Chinese/English OCR Visualization](#超轻量级中文OCR效果展示)
    - [General Chinese/English OCR Visualization](#通用中文OCR效果展示)
    - [General Chinese/English OCR Visualization (Support Space Recognization )](#支持空格的中文OCR效果展示)
- [TECHNICAL EXCHANGE GROUP](#WELCOME TO THE PaddleOCR TECHNICAL EXCHANGE GROUP)
- [REFERENCES](./doc/doc_en/reference_en.md)
- [LICENSE](#LICENSE)
- [CONTRIBUTION](#CONTRIBUTION)


## TEXT DETECTION ALGORITHM

PaddleOCR open source text detection algorithms list:
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))
- [ ]  SAST([paper](https://arxiv.org/abs/1908.05498))(Baidu Self-Research, comming soon)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[Download link](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[Download link](https://paddleocr.bj.bcebos.com/det_mv3_east.tar)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[Download link](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[Download link](https://paddleocr.bj.bcebos.com/det_mv3_db.tar)|

For use of [LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/datasets_en.md#1-icdar2019-lsvt) street view dataset with a total of 3w training data，the related configuration and pre-trained models for Chinese detection task are as follows:
|Model|Backbone|Configuration file|Pre-trained model|
|-|-|-|-|
|ultra-lightweight Chinese model|MobileNetV3|det_mv3_db.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|
|General Chinese OCR model|ResNet50_vd|det_r50_vd_db.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|

* Note: For the training and evaluation of the above DB model, post-processing parameters box_thresh=0.6 and unclip_ratio=1.5 need to be set. If using different datasets and different models for training, these two parameters can be adjusted for better result.

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./doc/doc_en/detection_en.md)

## TEXT RECOGNITION ALGORITHM

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [ ]  SRN([paper](https://arxiv.org/abs/2003.12294))(Baidu Self-Research, comming soon)

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|
|RARE|Resnet34_vd|84.90%|rec_r34_vd_tps_bilstm_attn|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_attn.tar)|
|RARE|MobileNetV3|83.32%|rec_mv3_tps_bilstm_attn|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_attn.tar)|

We use [LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/datasets_en.md#1-icdar2019-lsvt) dataset and cropout 30w  traning data from original photos by using position groundtruth and make some calibration needed. In addition, based on the LSVT corpus, 500w synthetic data is generated to train the Chinese model. The related configuration and pre-trained models are as follows:
|Model|Backbone|Configuration file|Pre-trained model|
|-|-|-|-|
|ultra-lightweight Chinese model|MobileNetV3|rec_chinese_lite_train.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) & [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)|
|General Chinese OCR model|Resnet34_vd|rec_chinese_common_train.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) & [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)|

Please refer to the document for training guide and use of PaddleOCR text recognition algorithms [Text recognition model training/evaluation/prediction](./doc/doc_en/recognition_en.md)

## END-TO-END OCR ALGORITHM
- [ ]  [End2End-PSL](https://arxiv.org/abs/1909.07808)(Baidu Self-Research, comming soon)

## Visualization

<a name="Ultra-lightweight Chinese/English OCR Visualization"></a>
### 1.Ultra-lightweight Chinese/English OCR Visualization  [more](./doc/doc_en/visualization.md)

<div align="center">
    <img src="doc/imgs_results/1.jpg" width="800">
</div>

<a name="General Chinese/English OCR Visualization"></a>
### 2. General Chinese/English OCR Visualization  [more](./doc/doc_en/visualization_en.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/11.jpg" width="800">
</div>

<a name=" General Chinese/English OCR Visualization(Space_support)"></a>
### 3.General Chinese/English OCR Visualization (Space_support ) [more](./doc/doc_en/visualization_en.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/en_paper.jpg" width="800">
</div>

<a name="FAQ"></a>

## FAQ
1. Error when using attention-based recognition model: KeyError: 'predict'

    The inference of recognition model based on attention loss is still being debugged. For Chinese text recognition, it is recommended to choose the recognition model based on CTC loss first. In practice, it is also found that the recognition model based on attention loss is not as effective as the one based on CTC loss.

2. About inference speed

    When there are a lot of texts in the picture, the prediction time will increase. You can use `--rec_batch_num` to set a smaller prediction batch size. The default value is 30, which can be changed to 10 or other values.

3. Service deployment and mobile deployment

    It is expected that the service deployment based on Serving and the mobile deployment based on Paddle Lite will be released successively in mid-to-late June. Stay tuned for more updates.

4. Release time of self-developed algorithm

    Baidu Self-developed algorithms such as SAST, SRN and end2end PSL will be released in June or July. Please be patient.

[more](./doc/doc_en/FAQ_en.md)

## WELCOME TO THE PaddleOCR TECHNICAL EXCHANGE GROUP
Scan  the QR code below with your wechat and completing the questionnaire, you can access to offical technical exchange group

<div align="center">
<img src="./doc/joinus.jpg"  width = "200" height = "200" />
</div>

## REFERENCES
```
1. EAST:
@inproceedings{zhou2017east,
  title={EAST: an efficient and accurate scene text detector},
  author={Zhou, Xinyu and Yao, Cong and Wen, He and Wang, Yuzhi and Zhou, Shuchang and He, Weiran and Liang, Jiajun},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={5551--5560},
  year={2017}
}

2. DB:
@article{liao2019real,
  title={Real-time Scene Text Detection with Differentiable Binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  journal={arXiv preprint arXiv:1911.08947},
  year={2019}
}

3. DTRB:
@inproceedings{baek2019wrong,
  title={What is wrong with scene text recognition model comparisons? dataset and model analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4715--4723},
  year={2019}
}

4. SAST:
@inproceedings{wang2019single,
  title={A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning},
  author={Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={1277--1285},
  year={2019}
}

5. SRN:
@article{yu2020towards,
  title={Towards Accurate Scene Text Recognition with Semantic Reasoning Networks},
  author={Yu, Deli and Li, Xuan and Zhang, Chengquan and Han, Junyu and Liu, Jingtuo and Ding, Errui},
  journal={arXiv preprint arXiv:2003.12294},
  year={2020}
}

6. end2end-psl:
@inproceedings{sun2019chinese,
  title={Chinese Street View Text: Large-scale Chinese Text Reading with Partially Supervised Learning},
  author={Sun, Yipeng and Liu, Jiaming and Liu, Wei and Han, Junyu and Ding, Errui and Liu, Jingtuo},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9086--9095},
  year={2019}
}
```

## LICENSE
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>

## CONTRIBUTION
We welcome all the contributions to PaddleOCR and appreciate for your feedback very much.

- Many thanks to [Khanh Tran](https://github.com/xxxpsyduck) for contributing the English documentation.
- Many thanks to [zhangxin](https://github.com/ZhangXinNan) for contributing the new visualize function、add .gitgnore and discard set PYTHONPATH manually.
- Many thanks to [lyl120117](https://github.com/lyl120117) for contributing the code for printing the network structure.
