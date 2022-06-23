# Training Command
python train.py \
--do_train \
--group_name reproduction \
--data_path QuoQA-NLP/train-only \
--use_validation False \
--PLM klue/roberta-large \
--model_category roberta \
--model_name RobertaForV2QuestionAnswering \
--max_length 460 \
--stride 128 \
--save_strategy steps \
--save_steps 125 \
--overwrite_output_dir \
--save_total_limit 10 \
--output_dir ./exps \
--logging_dir ./logs \
--learning_rate 2e-5 \
--weight_decay 2e-2 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--warmup_ratio 0.10 \
--max_steps 1500 \
--seed 42



# python inference.py \
# --do_predict \
# --PLM checkpoints \
# --model_category roberta \
# --model_name RobertaForV2QuestionAnswering \
# --max_length 512 \
# --stride 128 \
# --output_dir results \
# --overwrite_output_dir \
# --per_device_eval_batch_size 16 \
# --file_name roberta_base_EP:5_BS:16_WR:0.05_WD:1e-3.csv