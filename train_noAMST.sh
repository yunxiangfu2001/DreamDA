torchrun --nproc_per_node=8 --master_port 29507 train.py \
--dataset pet \
--data-dir /home/h3571902/data/pet_data \
--synthetic \
--synthetic-data-dir synthetic_data_generation/data/synthetic/pet/ \
--train-split 'dreamda_x30_nofinetune' \
--val-split 'test' \
--model resnet50 \
--lr 0.01 --warmup-epochs 5 --epochs 1000 --weight-decay 1e-4 --sched cosine \
-b 8 \
--num-classes 37 \
--output 'output/pet/dreamda_x30_resnet50_pretrained' \
--seed 0 --pretrained

# /path/to/synthetic/data/