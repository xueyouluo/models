export GLUE_DIR=/data/xueyou/data/corpus/task_data/lcqmc
export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_L-12_H-768_A-12

export TASK_NAME=QQP
export OUTPUT_DIR=${GLUE_DIR}/dataset
python ../data/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}