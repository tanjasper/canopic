ROOT_DATA_DIR='data/results'
VGGFACE2_DIR='data/vggface2'

SAVE_DIR='Learned_Conv1_Maxpool_Quantize/alpha50/id_classification'
IM_DIM=64
TR_FILENAMES='shared_filenames/identity_classification/vggface2_test_100identities_splitX-30-30_train_filenames.txt'
VAL_FILENAMES='shared_filenames/identity_classification/vggface2_test_100identities_splitX-30-30_val_filenames.txt'
TEST_FILENAMES='shared_filenames/identity_classification/vggface2_test_100identities_splitX-30-30_test_filenames.txt'
ENCODER_CHECKPOINT_PATH='Learned_Conv1_Maxpool_Quantize/alpha50/encoder_checkpoint_epoch18_exactquantize.tar'
DISCRIMINATOR_SETTINGS_JSON='json_templates/id_classification/googlenetIN_3ch_discriminator.json'

# TRAINING PARAMETERS
WAIT_EPOCHS=40
LR='0.001'
NUM_LRS='3'

# copy this script over to the save directory. also log current git commit.
mkdir -p $ROOT_DATA_DIR/$SAVE_DIR
SHELL_PATH=$(readlink -f "$0")
cp $SHELL_PATH $ROOT_DATA_DIR/$SAVE_DIR/'run_command.sh'
git show --oneline -s > $ROOT_DATA_DIR/$SAVE_DIR/'git_commit_info.txt'

python3 identity_classification.py \
--save_dir $SAVE_DIR \
--im_dim $IM_DIM \
--tr_filenames $TR_FILENAMES --val_filenames $VAL_FILENAMES --test_filenames $TEST_FILENAMES \
--imdir $VGGFACE2_DIR --root_data_dir $ROOT_DATA_DIR \
--wait_epochs $WAIT_EPOCHS --lr $LR --num_lrs $NUM_LRS \
--encoder_checkpoint_path $ENCODER_CHECKPOINT_PATH \
--discriminator_settings_json $DISCRIMINATOR_SETTINGS_JSON
