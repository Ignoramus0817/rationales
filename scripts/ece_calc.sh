model_path=
task_name=
test_data=
train_data=
output_file=
prompt_source=

# w/ rationale
CUDA_VISIBLE_DEVICES=0 nohup python ece_calc_other.py \
    --model_path ${model_path} \
    --task ${task_name} \
    --test_data ${test_data} \
    --output_file ${output_file} \
    --train_data ${train_data} \
    --rationale \
    --demo_num 0 \
    --info_file ${prompt_source} \
    > log 2>&1 &

# wo/ rationale
CUDA_VISIBLE_DEVICES=0 nohup python ece_calc_other.py \
    --model_path ${model_path} \
    --task ${task_name} \
    --test_data ${test_data} \
    --output_file ${output_file} \
    --train_data ${train_data} \
    --demo_num 0 \
    --info_file data/dataset_info.json \
    > log 2>&1 &
