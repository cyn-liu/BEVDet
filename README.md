# A fork of BEVDet

This repository mainly provides training/evaluation of TIER4 dataset and the script for exporting onnx required by [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp/tree/one).

## Environment
Please refer to [BEVDet](https://github.com/HuangJunJie2017/BEVDet/tree/dev2.1).

## Data Prepration
Prepare TIER4 dataset, download dataset from [Odaiba_JT_v1.0.zip](https://drive.google.com/file/d/1UaarK88HZu09sf7Ix-bEVl9zGNGFwTVL/view?usp=sharing), extract the dataset and place it in the `BEVDet/data` directory. 

Divide the TIER4 dataset into a training set and a validation set, and write the results to `Odaiba_JT_v1.0.yaml`:

```yaml
train:
  - 1cf17b50-551f-4597-b589-01edf4b1302a
  #- 1e4f67df-a759-4597-add2-d89cf2bade2d
  - 2f16dac8-a275-4780-98ca-9de6ca746367
  - 3dd8241a-83da-44dc-98c6-1a62beec217a
  - 8fffd37e-2296-4b00-a13a-a14de763c7c0
  - 9d6dfb10-d605-4243-a1f2-547c82a15caf
  - 37d8e37d-4311-47ce-a0d1-6e777e35ae39
  - 38da8b94-8ba4-4ac1-a24a-51dccdb5c6b5
  - 043f584e-dee3-4f29-997f-8a15e87284dd
  - 76eb738b-a28a-4a66-90b2-e08cd35a850a
  - 80fdca75-6b5d-490a-9457-0106e43183a1
  - 95c5cc92-3e2a-45a4-aba1-1c24b33a95fb
  - 98e82228-4cac-4b19-95e2-66f5325f7511
  - 595f2754-ebcc-44f1-9e8f-a0620392cb75
  - 2712d551-860d-4bc1-b12d-a2cd89e037a2
  - 7401dd84-8ff3-4ba7-a060-dbb9601f9483
  - 7735d716-7f2c-4d6c-b823-c690c4099204
  - 725282a0-7a1e-47ff-bf0d-01fcf93760cc
  - a97ea52c-bd31-465a-920e-8fc47a6bc574
  - a06442a3-dd46-4925-b2a1-3d465cdd860f
  - ae31b14c-1720-4466-90f1-bddc53ca560a
  - b3c27f97-224d-4a6c-bf2b-2c5e973ece5c
  - b90db8b1-896f-4152-add1-636ca00de46a
  - b91f8b04-965f-4c7c-87e8-487277cfc4b3
  - b449e9eb-daeb-4934-89c7-c97ddbc3a615
  - b3746b9e-c9f4-4bf9-98ea-d0d77351adf5
  - bbe2e907-df55-484d-a5b9-09030dbc5c13
  - d5bb1c0e-b27b-4ab6-b677-6a24d5b994f8
  - df2c169d-ffc0-4793-9978-63b72151675e
  - e5b2bc10-8c4f-4990-9eeb-f31e6791227a
  - ec95e7e6-a2bd-4335-9923-bd90fe0c0b13
  - f63a62f3-c274-49e4-8ab8-6683a93135f5
val:
  - 2a1b95d2-52c0-4ede-9980-496f1d2c3727
  - 11eb2027-85ec-4836-a6ca-f43d50216bd4
  - 34a0f8d1-95a5-4b10-bc4f-83672c115189
  - 0622e803-db00-48af-8d8d-dbc90263bce4
  - 3822b30f-3a7f-4d6a-9177-607eb4cb649c
  - b572345a-7b35-43fb-87ba-402215eba8e4
  - bee54781-559c-418d-94aa-b0ad005d6dfd
test:

```

Create the pkl for BEVDet by running:

```shell
python tools/create_tier4_data_bevdet.py
```

Finally, `Odaiba_JT_v1.0` folder was arrange as:

```shell script
Odaiba_JT_v1.0/
│
├── 043f584e-dee3-4f29-997f-8a15e87284dd/
│   ├── annotation
│   └── data
│
├── 2f16dac8-a275-4780-98ca-9de6ca746367/
│   ├── annotation
│   └── data
│
└── Odaiba_JT_v1.0.yaml
│
└── bevdetv2-tier4_infos_train.pkl
│
└── bevdetv2-tier4_infos_val.pkl
│
└── bevdetv2-tier4_infos_test.pkl

```

## Train model

```shell
# single gpu
python tools/train.py configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py
# multiple gpu
./tools/dist_train.sh configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py $num_gpu
```

## Test model

```shell
# single gpu
python tools/test.py configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py $checkpoint $num_gpu --eval mAP
```

## Export Onnx

export onnx:

uncomment `line524` in file`mmdet3d/models/neck/view_transformer.py` and comment `line523` in file `mmdet3d/models/neck/view_transformer.py`.

```shell
python tools/export/export_one_onnx.py configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py $checkpoint --postfix='_lt_d' 
```

export yaml:

```shell
python tools/export/export_yaml.py configs/bevdet/tier4dataset-bevdet-r50-4dlongterm-depth-cbgs.py --prefix='bevdet_lt_d'
```