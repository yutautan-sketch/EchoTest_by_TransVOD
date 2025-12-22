python util/vid_2_frames.py\
     --interval 2\
     --rm_exdir

python infer_wo_anno.py\
     --resume results/transvod/251130/femur_251128_mm=0.0_topk=3_boxes=gt/checkpoint0049.pth\
     --backbone swin_s_p4w7\
     --batch_size=1\
     --num_feature_levels 1\
     --num_queries 12\
     --num_frames 12\
     --with_box_refine\
     --dilation\
     --output_dir results/wo_anno_femur_frame\
     --dataset_file vid_multi\
     --enc_layers 3\
     --dec_layers 3\
     --val_path ./data/wo_anno