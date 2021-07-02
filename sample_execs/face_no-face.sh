# Change the following directories accordingly
ROOT_DATA_DIR='data/results'
ILSVRC2012_DIR='data/ILSVRC2012/images'
VGGFACE2_DIR='data/vggface2'

SAVE_DIR='Learned_Conv1_Maxpool_Quantize/alpha50/face_no-face'
IM_DIM=64
TR_FILENAMES_FACE='shared_filenames/face_noface_classification/jumbled_vggface2_training_filenames_1-100k.txt'
VAL_FILENAMES_FACE='shared_filenames/face_noface_classification/jumbled_vggface2_training_filenames_100001-110k.txt'
TEST_FILENAMES_FACE='shared_filenames/face_noface_classification/vggface2_test_100identities_splitX-30-30_test_filenames.txt'
TR_FILENAMES_NOFACE='shared_filenames/face_noface_classification/ilsvrc_train_faceless_100k_from138k.txt'
VAL_FILENAMES_NOFACE='shared_filenames/face_noface_classification/ilsvrc_train_faceless_10k_from138k.txt'
TEST_FILENAMES_NOFACE='shared_filenames/face_noface_classification/jumbled_ilsvrc_val_faceless_filenames_1-3k.txt'
ENCODER_CHECKPOINT_PATH='Learned_Conv1_Maxpool_Quantize/alpha50/encoder_checkpoint_epoch18_exactquantize.tar'
DISCRIMINATOR_SETTINGS_JSON='json_templates/face_noface/googlenetIN_3ch_discriminator.json'

# TRAINING PARAMETERS
WAIT_EPOCHS=5
LR='0.001'
NUM_LRS='3'

# copy this script over to the save directory. also log current git commit.
mkdir -p $ROOT_DATA_DIR/$SAVE_DIR
SHELL_PATH=$(readlink -f "$0")
cp $SHELL_PATH $ROOT_DATA_DIR/$SAVE_DIR/'run_command.sh'
git show --oneline -s > $ROOT_DATA_DIR/$SAVE_DIR/'git_commit_info.txt'

python3 face_noface_classification.py \
--save_dir $SAVE_DIR \
--im_dim $IM_DIM \
--tr_filenames_face $TR_FILENAMES_FACE --val_filenames_face $VAL_FILENAMES_FACE --test_filenames_face $TEST_FILENAMES_FACE \
--tr_filenames_noface $TR_FILENAMES_NOFACE --val_filenames_noface $VAL_FILENAMES_NOFACE --test_filenames_noface $TEST_FILENAMES_NOFACE \
--imdir_face $VGGFACE2_DIR --imdir_noface $ILSVRC2012_DIR --root_data_dir $ROOT_DATA_DIR \
--wait_epochs $WAIT_EPOCHS --lr $LR --num_lrs $NUM_LRS \
--encoder_checkpoint_path $ENCODER_CHECKPOINT_PATH \
--discriminator_settings_json $DISCRIMINATOR_SETTINGS_JSON

