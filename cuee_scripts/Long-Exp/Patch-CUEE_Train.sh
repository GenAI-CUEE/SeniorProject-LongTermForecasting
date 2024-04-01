if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=4 
label_len=0
moving_avg=37 
batch_size=128
target=I
seq_len=74
model_name=PatchTST

feature_type=MS 
num_features=8 
patch_len=16
stride=8

d_model=128

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE/ \
    --data_path updated_measurement_Iclr_new.csv \
    --model_id CUEEData_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
    --data CUEE \
    --features $feature_type \
    --target $target \
    --seq_len $seq_len \
    --label_len $label_len\
    --pred_len $pred_len \
    --enc_in $num_features \
    --e_layers 3 \
    --n_heads 4 \
    --d_model $d_model \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride \
    --des 'Exp' \
    --train_epochs 10\
    --patience 3\
    --lradj 'TST'\
    --pct_start 0.3\
    --itr 1 --batch_size $batch_size --learning_rate 0.0001 >'logs/LongForecasting/'$model_name"_"$feature_type"_d"$d_model"_"$patch_len"-"$stride"_"$target"_mv"$moving_avg"_CUEE_"$seq_len'_'$label_len'_'$pred_len'_'$batch_size'.log' 