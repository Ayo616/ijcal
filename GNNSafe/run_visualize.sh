
dev=1

# output and store energy scores for visualization

#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_reg --lamda 0.1 --m_in -7 --m_out -2 --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_prop --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_prop --use_reg --lamda 0.1 --m_in -5 --m_out -1 --use_bn --device $dev
#
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_reg --lamda 0.1 --m_in -9 --m_out -4 --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_prop --use_bn --device $dev
#python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_prop --use_reg --lamda 0.1 --m_in -9 --m_out -2 --use_bn --device $dev


# yan
#python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset twitch --mode detect --use_bn --device $dev
#python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset twitch --mode detect --use_bn  --device 1 --lamda 0.5  --use_mlayer  --record Model_mlayer
#python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset twitch --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1 --record Model_slayer  --num_layers 5
#python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset twitch --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 0.8 --record Model_Final

python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset arxiv --mode detect --use_bn --device $dev
python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset arxiv --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset arxiv --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 0.9 --record Model_slayer
python discuss2.py --dis_type vis_energy --method GEVN --backbone gcn --dataset arxiv --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 0.9 --record Model_Final

