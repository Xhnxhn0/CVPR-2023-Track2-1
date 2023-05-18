python train.py \
--name iira \
--img_aug \
--batch_size 90 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name 'MY-DATA' \
--root_dir './' \
--weight_decay 1e-3 \
--lr 1e-5 