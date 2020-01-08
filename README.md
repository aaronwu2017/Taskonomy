# Taskonomy
Transfer Taskonomy to novel Tasks with multi-output decoder.<br />
resnet_model.py is the baseline method for apparel classification. It contains a resnet-50 used as encoder.<br />
Load_data.py is to process data<br />
test_all_model can save loss in a CSV file<br />

sample code to mass produce the representations: python  tools/run_img_task2.py --task class_1000 --input data/dataset/red_dress/names.txt  --output  data/dataset/red_dress/class_1000
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0001.jpg)
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0002.jpg)
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0003.jpg)
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0004.jpg)
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0005.jpg)
![alt text](https://github.com/aaronwu2017/Taskonomy/blob/master/cs5070%20final%20draft/0006.jpg)
