# RanDumb -- All Runs
python main.py --embed --embed_dim 1000 --model SLDA --dataset mnist
python main.py --embed --embed_dim 1000 --model SLDA --dataset cifar10
python main.py --embed --embed_dim 1000 --model SLDA --dataset cifar100
python main.py --embed --embed_dim 1000 --augment --model SLDA --dataset cifar10
python main.py --embed --embed_dim 1000 --augment --model SLDA --dataset cifar100

python main.py --embed --embed_dim 1000 --model NCM --dataset mnist
python main.py --embed --embed_dim 1000 --model NCM --dataset cifar10
python main.py --embed --embed_dim 1000 --model NCM --dataset cifar100
python main.py --embed --embed_dim 1000 --augment --model NCM --dataset cifar10
python main.py --embed --embed_dim 1000 --augment --model NCM --dataset cifar100

python main.py --embed --embed_dim 2000 --model SLDA --dataset mnist
python main.py --embed --embed_dim 2000 --model SLDA --dataset cifar10
python main.py --embed --embed_dim 2000 --model SLDA --dataset cifar100
python main.py --embed --embed_dim 2000 --augment --model SLDA --dataset cifar10
python main.py --embed --embed_dim 2000 --augment --model SLDA --dataset cifar100

python main.py --embed --embed_dim 2000 --model NCM --dataset mnist
python main.py --embed --embed_dim 2000 --model NCM --dataset cifar10
python main.py --embed --embed_dim 2000 --model NCM --dataset cifar100
python main.py --embed --embed_dim 2000 --augment --model NCM --dataset cifar10
python main.py --embed --embed_dim 2000 --augment --model NCM --dataset cifar100