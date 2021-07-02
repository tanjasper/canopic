# CAnOPIC

This is the official repository for the paper: "CANOPIC: Pre-Digital Privacy-Enhancing Encodings for Computer Vision".
This repository has been tested with pytorch1.6 and python3.7.7.

Paper: https://ieeexplore.ieee.org/abstract/document/9102956

## Data

You will need to download ILSVRC2012 and vggface2 into the `data` directory. You should have the folders `data/ILSVRC2012/images/train`, `data/ILSVRC2012/images/val`, `data/vggface2/train`, and `data/vggface2/test`. Alternatively, you can place these folders in different directories, but you will need to change the data directory arguments in the different scripts. Copy `data/ILSVRC2012/images/filenames` and `data/vggface2/filenames` into your directories for ILSVRC2012 and vggface2.

## Training a CAnOPIC

The parameters for the CAnOPIC (encoder) such as what operations it includes (conv, maxpool, quantize, etc.) are passed in using json files. See `json_templates/train_encoder/encoders` for examples.

The parameters for the neural networks are also passed in using json files. See `json_templates/train_encoder/discriminators` for example.

Examples:

To train a Conv(1)-Maxpool-Quantize CAnOPIC with alpha=50:
```
python train_encoder.py --save_dir Learned_Conv4_Maxpool_Quantize --results_dir alpha50 --encoder_settings_json json_templates/train_encoder/encoders/encoder_conv4_maxpool_quantize.json --entropy_weight 50
```
The `train_encoder.py` script also contains a number of other parameters. The default parameters set are the parameters used for the paper.

To train a Conv(4)-Maxpool CAnOPIC with alpha=300:
```
python train_encoder.py --save_dir Learned_Conv1_Maxpool --results_dir alpha300 --encoder_settings_json json_templates/train_encoder/encoders/encoder_Convolution_MaxPool.json --entropy_weight 300 --discriminatorA_settings_json json_templates/train_encoder/discriminators/resnet18IN_100class_in12_settings.json --discriminatorB_settings_json json_templates/train_encoder/discriminators/resnet18IN_2class_in12_settings.json
```
Note that for Conv(4), we need to use a discriminator that can take in 12 channels as inputs. The default discriminator is one set to take in 3 channels as inputs. Thus, we have to pass in different `--discriminatorA_settings_json` and `--discriminatorB_settings_json` for the Conv(4) case.

## Testing the CAnOPIC (Obtaining Table 1 in paper)

A sample script to obtain the ID classification score for a single CAnOPIC can be found in `sample_execs/id_classification.sh`. This particular one is for the CAnOPIC saved in `data/results/Learned_Conv1_Maxpool_Quantize/weight50/encoder_checkpoint_epoch18_exactquantize.tar`. To run this on a different CAnOPIC, change the `ENCODER_CHECKPOINT_PATH` variable to the path your CAnOPIC checkpoint is saved at. Note that this path is relative to `ROOT_DATA_DIR`.

A sample script to obtain the Face vs. No-Face score for a single CAnOPIC can be found in `sample_execs/face_no-face.sh`. Again, this is for the CAnOPIC saved in `data/results/Learned_Conv1_Maxpool_Quantize/weight50/encoder_checkpoint_epoch18_exactquantize.tar`.

For both examples, when running on the Conv(4) case, you will need to change the `DISCRIMINATOR_SETTINGS_JSON` to `json_templates/id_classification/googlenetIN_disccriminator.json`.
