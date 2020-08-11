export BERT_DIR=/data/xueyou/data/bert_pretrain/converted/chinese_L-12_H-768_A-12
export MODEL_DIR=/data/xueyou/data/corpus/task_data/lcqmc/bert
export GLUE_DIR=/data/xueyou/data/corpus/task_data/lcqmc/dataset
export TASK=QQP

python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=${GLUE_DIR}/${TASK}_meta_data \
  --train_data_path=${GLUE_DIR}/${TASK}_train.tf_record \
  --eval_data_path=${GLUE_DIR}/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_DIR}/bert_config.json \
  --init_checkpoint=${BERT_DIR}/bert_model-1 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --steps_per_loop=1 \
  --learning_rate=2e-5 \
  --num_eval_per_epoch=4 \
  --num_train_epochs=3 \
  --model_dir=${MODEL_DIR} \
  --distribution_strategy=mirrored