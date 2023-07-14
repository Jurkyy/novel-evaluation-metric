# install python dependencies
#pip3 install -r "$(dirname "$0")"/requirements.txt

################################################################################################################################################################################################
###################################################################################### download datasets #######################################################################################
################################################################################################################################################################################################

mkdir "$(dirname "$0")"/datasets/{AES_HD,AES_RD,ASCAD/0,ASCAD/50,ASCAD/100,DPAv4}/training_data

# AES_HD
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/AES_HD/AES_HD_dataset.zip -o "$(dirname "$0")"/datasets/AES_HD/training_data/AES_HD_dataset.zip
unzip "$(dirname "$0")"/datasets/AES_HD/training_data/AES_HD_dataset -d  "$(dirname "$0")"/datasets/AES_HD/training_data/
rm "$(dirname "$0")"/datasets/AES_HD/training_data/AES_HD_dataset.zip
mv "$(dirname "$0")"/datasets/AES_HD/training_data/*/* "$(dirname "$0")"/datasets/AES_HD/training_data/ && rm -rf "$(dirname "$0")"/datasets/AES_HD/training_data/AES_HD_dataset

# AES_RD
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/AES_RD/AES_RD_dataset/AES_RD_attack.zip -o "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_attack.zip
unzip "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_attack.zip -d "$(dirname "$0")"/datasets/AES_RD/training_data/
rm "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_attack.zip

curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/AES_RD/AES_RD_dataset/AES_RD_profiling.zip -o "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_profiling.zip
unzip "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_profiling.zip  -d "$(dirname "$0")"/datasets/AES_RD/training_data/
rm "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_profiling.zip

curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/AES_RD/AES_RD_dataset/key.npy -o "$(dirname "$0")"/datasets/AES_RD/training_data/key.npy

mv "$(dirname "$0")"/datasets/AES_RD/training_data/*/* "$(dirname "$0")"/datasets/AES_RD/training_data/
rm -rf "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_attack && rm -rf "$(dirname "$0")"/datasets/AES_RD/training_data/AES_RD_profiling

# ASCAD_0
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/ASCAD/N0%3D0/ASCAD_dataset.zip -o "$(dirname "$0")"/datasets/ASCAD/0/training_data/ASCAD_0_dataset.zip
unzip "$(dirname "$0")"/datasets/ASCAD/0/training_data/ASCAD_0_dataset.zip -d "$(dirname "$0")"/datasets/ASCAD/0/training_data/
rm "$(dirname "$0")"/datasets/ASCAD/0/training_data/ASCAD_0_dataset.zip
mv "$(dirname "$0")"/datasets/ASCAD/0/training_data/ASCAD_dataset/* "$(dirname "$0")"/datasets/ASCAD/0/training_data/ && rm -rf "$(dirname "$0")"/datasets/ASCAD/0/training_data/ASCAD_dataset

# ASCAD_50
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/ASCAD//N0%3D50/ASCAD_dataset.zip -o "$(dirname "$0")"/datasets/ASCAD/50/training_data/ASCAD_50_dataset.zip
unzip "$(dirname "$0")"/datasets/ASCAD/50/training_data/ASCAD_50_dataset.zip -d "$(dirname "$0")"/datasets/ASCAD/50/training_data/
rm "$(dirname "$0")"/datasets/ASCAD/50/training_data/ASCAD_50_dataset.zip
mv "$(dirname "$0")"/datasets/ASCAD/50/training_data/ASCAD_dataset/* "$(dirname "$0")"/datasets/ASCAD/50/training_data/ && rm -rf "$(dirname "$0")"/datasets/ASCAD/50/training_data/ASCAD_dataset

# ASCAD_100
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/ASCAD/N0%3D100/ASCAD_dataset.zip -o "$(dirname "$0")"/datasets/ASCAD/100/training_data/ASCAD_100_dataset.zip
unzip "$(dirname "$0")"/datasets/ASCAD/100/training_data/ASCAD_100_dataset.zip -d "$(dirname "$0")"/datasets/ASCAD/100/training_data/
rm "$(dirname "$0")"/datasets/ASCAD/100/training_data/ASCAD_100_dataset.zip
mv "$(dirname "$0")"/datasets/ASCAD/100/training_data/ASCAD_dataset/* "$(dirname "$0")"/datasets/ASCAD/100/training_data/ && rm -rf "$(dirname "$0")"/datasets/ASCAD/100/training_data/ASCAD_dataset

# DPA-contest_v4
curl -L https://raw.githubusercontent.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/master/DPA-contest%20v4/DPAv4_dataset.zip -o "$(dirname "$0")"/datasets/DPAv4/training_data/DPAv4_dataset.zip
unzip "$(dirname "$0")"/datasets/DPAv4/training_data/DPAv4_dataset.zip -d "$(dirname "$0")"/datasets/DPAv4/training_data/
rm "$(dirname "$0")"/datasets/DPAv4/training_data/DPAv4_dataset.zip
mv "$(dirname "$0")"/datasets/DPAv4/training_data/*/* "$(dirname "$0")"/datasets/DPAv4/training_data/ && rm -rf "$(dirname "$0")"/datasets/DPAv4/training_data/DPAv4_dataset