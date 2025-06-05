export CUDA_VISIBLE_DEVICES=4,5,6,7
task=winogrande
rationale=True/False
ckpt_save_dir=/output/path

nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 train/train.py \
    --model_name_or_path /path/to/your/model/ \
    --data_path /path/to/synthesized/data \
    --task_name ${task} \
    --rationale ${rationale} \
    --bf16 True \
    --output_dir ${ckpt_save_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --report_to "tensorboard" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    > log.txt 2>&1 &