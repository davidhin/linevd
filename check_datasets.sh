feat="_ABS_DATAFLOW_datatype_all"

./mypython code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --batch_size 256 --train_workers 6 --max_epochs 250 --weight_decay 1e-2 \
    --label_style graph --split fixed \
    --seed 0 --skip_train &> splits_fixed.txt

./mypython code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --batch_size 256 --train_workers 6 --max_epochs 250 --weight_decay 1e-2 \
    --label_style graph --split random \
    --seed 0 --skip_train &> splits_random.txt
