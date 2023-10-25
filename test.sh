#qgm=1 is quasi global momentum and qgm=0 is local momentum
# CCL is activate when either lambda_m or lambda_d is non-zero

# DSGDm-N for CIFAR-10 on ResNet-20,  ring with 16 agents, skew 0.01.
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0  --lambda_d=0 --epoch=200 --arch=resnet --momentum=0.9 --qgm=0 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --depth=20 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0 --arch=resnet --world_size=16  --skew=0.01 --graph=ring
cd ..

# QG-DSGDm-N for CIFAR-10 on ResNet-20,  ring with 16 agents, skew 0.01.
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0  --lambda_d=0 --epoch=200 --arch=resnet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --depth=20 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0 --arch=resnet --world_size=16  --skew=0.01 --graph=ring
cd ..

# CCL with QGM for CIFAR-10 on ResNet-20,  ring with 16 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=resnet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --depth=20 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=resnet --world_size=16  --skew=0.01 --graph=ring
cd ..

# CCL with QGM for CIFAR-10 on ResNet-20,  dyck with 32 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01 --gamma=0.9 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=resnet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=dyck --neighbors=3 --depth=20 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=resnet --world_size=32  --skew=0.01 --graph=dyck
cd ..

# CCL with QGM for CIFAR-10 on ResNet-20,  torus with 32 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=1024  --world_size=32 --skew=0.01 --gamma=0.9 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=resnet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=torus --neighbors=4 --depth=20 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=resnet --world_size=32  --skew=0.01 --graph=torus
cd ..

# CCL with QGM for CIFAR-100 on ResNet-20,  ring with 16 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=resnet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --depth=20 --dataset=cifar100 --classes=100
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=resnet --world_size=16  --skew=0.01 --graph=ring
cd ..

# CCL with QGM for Fashion MNIST on LeNet-5,  ring with 16 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=lenet5 --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --dataset=fmnist --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=lenet5 --world_size=16  --skew=0.01 --graph=ring
cd ..

# CCL with QGM for Imagenette on MobileNet-V2,  ring with 16 agents, skew 0.01
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epoch=200 --arch=mobilenet --momentum=0.9 --qgm=1 --seed=123 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --dataset=imagenette --classes=10
cd ./outputs
python dict_to_csv.py --norm=evonorm  --lr=0.1 --lambda_m=0.1 --arch=mobilenet --world_size=16  --skew=0.01 --graph=ring
cd ..