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
seq_len=24
target=I


# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=8 

model_name=Informer # Transformer # Autoformer
mode=val # test val 

for d_model in 8 16 32 64 128 256 512
do
    python -u plot_longExp.py \
        --mode $mode \
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
        --embed_type 4 \
        --e_layers 2 \
        --d_model $d_model \
        --d_layers 1 \
        --factor 3 \
        --enc_in $num_features \
        --dec_in $num_features \
        --c_out  $num_features \
        --des 'Exp' \
        --batch_size $batch_size --itr 1 >'logs/LongForecasting/'$model_name"_"$feature_type"_d"$d_model"_"$target"_mv"$moving_avg"_CUEE_"$seq_len'_'$label_len'_'$pred_len'_'$batch_size'_infer.log' 
done