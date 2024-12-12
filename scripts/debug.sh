
# model_name=TransformerGym
# python3 -m debugpy --listen localhost:8888 --wait-for-client \
#         run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Monthly' \
#   --model_id m4_Monthly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 512 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

model_name=TransformerGym_None_series-patching_sparse-attention
python3 -m debugpy --listen localhost:8888 --wait-for-client \
        run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'