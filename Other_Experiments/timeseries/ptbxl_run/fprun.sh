export CUDA_VISIBLE_DEVICES=0,1
cd /nfs/hpc/share/pourmans/timeseries-ggd/Medformer

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /nfs/hpc/share/pourmans/timeseries-ggd/Medformer/PTB-XL \
  --model_id PTB-XL-Indep-fp \
  --model Medformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 256 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations jitter0.2,scale0.2,drop0.5 \
  --swa \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --devices 0,1 \
  --use_ggd 0