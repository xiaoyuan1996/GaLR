cd ../..
CARD=1

CUDA_VISIBLE_DEVICES=$CARD python train.py --path_opt option/RSICD_mca/RSICD_GaLR.yaml

CUDA_VISIBLE_DEVICES=$CARD python test_ave.py --path_opt option/RSICD_mca/RSICD_GaLR.yaml
