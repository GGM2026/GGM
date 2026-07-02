

python train_cifar.py \
    --num_runs 1 \
    --master_seed 100 \
    --test_name "base_testing_F0" \
    --dataset "cifar10" \
    --width 32 \
    --height 32 \
    --in_channel 3 \
    --patch_size 4 \
    --inner_dim 192 \
    --transformer_layers 12 \
    --num_head 3 \
    --embed_dropout 0.0 \
    --attn_dropout 0.0 \
    --mlp_dropout 0.0 \
    --k_bits_x 1 \
    --k_bits_w 1 \
    --n_factor 1 \
    --rho_cap 0.99 \
    --batch_size 512 \
    --out_classes 10 \
    --epochs 450 \
    --learning_rate 0.0005 \
    --weight_decay 0.00
