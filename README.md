# Train_Transformers_CLIP
Finetune your CLIP(in transformers) on private datasets!


'''
python fine_tune_clip.py \
  --model_name "/path/to/pretrained/clip_model" \
  --save_path "./fine_tuned_model" \
  --train_json "/path/to/train_data.json" \
  --val_json "/path/to/val_data.json" \
  --num_epochs 50 \
  --batch_size 256 \
  --save_steps 200 \
  --loss_type siglip
'''
