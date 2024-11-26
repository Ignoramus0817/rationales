export CUDA_VISIBLE_DEVICES=
task=
rationale=
out_dir=
annotator=

nohup python -m torch.distributed.launch --nproc_per_node=NUM_GPU --master_port=PORT train_classification.py \
    --model_name_or_path PATH_TO_MODEL \
    --data_path PATH_TO_DATA \
    --task_name ${task} \
    --rationale ${rationale} \
    `#--rationale_sub pad` \
    `#--annotator ${annotator}` \
    --bf16 True \
    --output_dir ${out_dir} \
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
