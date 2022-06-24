# # all parameters
# python main.py \
# --batch \
# --board \
# --checkpoint \
# --data_dir \
# --epoch \
# --gpu \
# --K \
# --load \
# --load_pt \
# --lr \
# --predict \
# --save \
# --seed \
# --test \
# --train \

# # train and predict with K-Fold
# python main.py \
# --train --predict \
# --batch 8 \
# --epoch 3 \
# --data_dir data/ag_news_csv2 \
# --K 5 \
# --checkpoint 10 \
# --board --save --gpu 0 --lr 0.001 --seed 42 

# # train and test without K-Fold
# python main.py \
# --train --test\
# --batch 8 \
# --epoch 3 \
# --data_dir data/ag_news_csv2 \
# --checkpoint 10 \
# --board --save --gpu 0 --lr 0.001 --seed 42

# # load model
# --load checkpoint_0_epoch.pt \

python main.py \
--train --predict \
--batch 128 \
--epoch 10 \
--data_dir disaster \
--K 5 \
--checkpoint 10 \
--board --save --gpu 0 --lr 0.001 --seed 42