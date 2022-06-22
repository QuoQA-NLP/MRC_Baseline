# Training Command
python train.py \
--do_train \
--group_name reproduction \
--data_path QuoQA-NLP/train-only \
--use_validation False \
--PLM klue/roberta-large \
--model_category roberta \
--model_name RobertaForV2QuestionAnswering \
--max_length 450 \
--stride 128 \
--save_strategy steps \
--save_steps 250 \
--overwrite_output_dir \
--save_total_limit 10 \
--output_dir ./exps \
--logging_dir ./logs \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--max_steps 3000 \
--per_device_train_batch_size 32 \
--warmup_ratio 0.05 \
--weight_decay 1e-2 \
--gradient_checkpointing

python inference.py \
--do_predict \
--PLM checkpoints \
--model_category roberta \
--model_name RobertaForV2QuestionAnswering \
--max_length 512 \
--stride 128 \
--output_dir results \
--overwrite_output_dir \
--per_device_eval_batch_size 16 \
--file_name roberta_base_EP:5_BS:16_WR:0.05_WD:1e-3.csv