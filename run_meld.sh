python train.py \
  --lr=0.00007 \
  --dropout=0.11 \
  --l2=0.000075 \
  --batch-size=64 \
  --hidden_dim=512 \
  --n_head=8 \
  --epochs=1 \
  --windows=6 \
  --Dataset=MELD \
  --save_model_path=./MELD \
  #预热0轮         self.q_base = 0.2   self.lam = 0.9  self.p_exe = 0.2