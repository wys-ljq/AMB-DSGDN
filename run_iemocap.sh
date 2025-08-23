python leidatrain.py \
--lr=0.000068 \
--dropout=0.5 \
--l2=0.00005 \
--batch-size=16 \
--hidden_dim=512 \
--n_head=8 \
--epochs=100 \
--windows=20 \
--class-weight \
--Dataset=IEMOCAP \
--save_model_path=./IEMOCAP \

#  预热60轮         self.q_base = 0.3  self.lam = 0.9   self.p_exe = 0.5