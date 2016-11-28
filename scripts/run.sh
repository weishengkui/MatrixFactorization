unzip ../data/ml-1m.zip -d ../data/
nohup python ../src/mf.py > log.txt 2>&1 &
