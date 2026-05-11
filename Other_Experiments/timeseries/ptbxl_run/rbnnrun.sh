export CUDA_VISIBLE_DEVICES=0
cd /nfs/hpc/share/pourmans/Experiments/timeseries-ggd/Medformer

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /nfs/hpc/share/pourmans/Experiments/timeseries-ggd/Medformer/PTB-XL \
  --model_id PTB-XL-Indep-rbnn \
  --model Medformer \
  --data PTB-XL \
  --seq_len 250 \
  --e_layers 6 \
  --batch_size 256 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations jitter0.2,scale0.2,drop0.5 \
  --swa \
  --des Exp \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --devices 0 \
  --use_xnor 0 \
  --use_ggd 0 \
  --use_dorefa 0 \
  --use_rbnn 1