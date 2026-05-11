export CUDA_VISIBLE_DEVICES=0,1,2,3

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Dep \
  --model Reformer \
  --data ADFTD-Dependent \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --model Reformer \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep \
  --model Reformer \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --model Reformer \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB/ \
  --model_id PTB-Indep \
  --model Reformer \
  --data PTB \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Indep \
  --model Reformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 256 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10




python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UCI-HAR/ \
  --model_id UCI-HAR \
  --model Reformer \
  --data UCI-HAR \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FLAAP/ \
  --model_id FLAAP \
  --model Reformer \
  --data FLAAP \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10