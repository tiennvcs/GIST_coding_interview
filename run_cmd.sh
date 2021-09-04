# Statistic data
python statistic_dataset.py --input_dir data/train/

# Training model
## RNN
python train.py --data_dir ./data \
--model RNN --num_features 10000 --feature_dim 100 \
--test_size 0.2 --epochs 100 --batch_size 64 \
--work_dir ./runs/train/rnn/

## CNN
python train.py --data_dir ./data \
--model CNN --num_features 10000 --feature_dim 100 \
--test_size 0.2 --epochs 100 --batch_size 64 \
--work_dir ./runs/train/cnn/
