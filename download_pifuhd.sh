git clone https://github.com/facebookresearch/pifuhd
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
cd lightweight-human-pose-estimation.pytorch/
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
cd ../pifuhd/
sh ./scripts/download_trained_model.sh
