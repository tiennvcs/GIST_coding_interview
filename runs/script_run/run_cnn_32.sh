python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer adam \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_Adam_1e3_Adam_w/

python train.py --data_dir ./data --model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer adam \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_Adam_1e3_Adam_wo/

python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer sgd \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_SGD_1e3_w/

python train.py --data_dir ./data --model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer sgd \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_SGD_1e3_wo/

python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer rmsprop \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_RMSProp_1e3_w/

python train.py --data_dir ./data --model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-3 --optimizer rmsprop \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_RMSProp_1e3_wo/

python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer adam \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_Adam_1e2_Adam_w/

python train.py --data_dir ./data \--model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer adam \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_Adam_1e2_Adam_wo/

python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer sgd \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_SGD_1e2_w/

python train.py --data_dir ./data --model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer sgd \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_SGD_1e2_wo/

python train.py --data_dir ./data --model CNN \
--pretrain_embedding data/all.review.vec.txt \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer rmsprop \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_RMSProp_1e2_w/

python train.py --data_dir ./data --model CNN \
--num_words 10000 --feature_dim 100 \
--learning_rate 1e-2 --optimizer rmsprop \
--epochs 100 --batch_size 32 \
--work_dir ./runs/train/RNN_64_RMSProp_1e2_wo/