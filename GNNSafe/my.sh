### Cora with structure ood

#python main2.py --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final

#### Cora with feature ood

#python main.py --method msp --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final

#### Cora with label ood
#
#python main.py --method msp --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset cora --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final

#
#### Amazon-photo with structure ood
#
#python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --device 1
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final

#
### Amazon-photo with feature ood
#
#python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final

#
#### Amazon-photo with label ood
#
#python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type label --mode detect --device 1
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### Coauthor with structure ood

#python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 0
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### Coauthor with feature ood

#python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### Coauthor with label ood

#python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1.1 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### Twitch
#python main2.py --method msp --backbone gcn --dataset twitch --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn  --dataset twitch --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn  --dataset twitch  --mode detect --use_bn  --device 1 --lamda 0.5  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn  --dataset twitch --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 1 --record Model_slayer  --num_layers 5
#python main2.py --method GEVN --backbone gcn  --dataset twitch --mode detect --use_bn  --device 1 --lamda 0.5 --oodloss --record Model_oodloss
python main2.py --method GEVN --backbone gcn  --dataset twitch --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 0.8 --record Model_Final


#### Arxiv
#python main.py --method msp --backbone gcn --dataset arxiv --mode detect --use_bn --device 1
#python main2.py --method GEVN --backbone gcn  --dataset arxiv --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn  --dataset arxiv --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn  --dataset arxiv --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 0.9 --record Model_slayer
#python main2.py --method GEVN --backbone gcn  --dataset arxiv --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn  --dataset arxiv --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 0.9 --record Model_Final


### wiki-cs with structure
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### wiki-cs with feature
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type feature --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### wiki-cs with label
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type label --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset wiki-cs --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### actor with structure
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### actor with feature
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type feature --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type feature --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### actor with label
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type label --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type label --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset actor --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final



### webkb with structure
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type structure --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type structure --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type structure --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type structure --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### webkb with feature
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type feature --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type feature --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type feature --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type feature --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final


### webkb with label
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type label --mode detect --use_bn --lamda 1 --device 1 --record Model
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type label --mode detect --use_bn  --device 1 --lamda 1  --use_mlayer  --record Model_mlayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type label --mode detect --use_bn  --device 1  --use_sprop --use_srece  --reduce 6 --record Model_slayer
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --record Model_oodloss
#python main2.py --method GEVN --backbone gcn --dataset webkb --ood_type label --mode detect --use_bn  --device 1 --lamda 0.01 --oodloss --use_mlayer --use_sprop --use_srece  --reduce 1.1 --record Model_Final



