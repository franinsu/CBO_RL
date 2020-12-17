rm runs/* -rf
# nohup ngrok http 6006 --log=stdout > ngrok.log &
nohup tensorboard --logdir=runs &

nohup python3 ./cloud_script.py continuous resnet 0 0 1 0 0 0 100 10 90 > out1.log &
nohup python3 ./cloud_script.py discrete tabular 0 0 1 0 0 0 100 10 90 > out2.log &

# nohup python3 ./cloud_script.py continuous resnet 0 0 1 0 0 0 0 5 90 > out1.log &
# nohup python3 ./cloud_script.py discrete tabular 0 0 1 0 0 0 0 5 90 > out2.log &
