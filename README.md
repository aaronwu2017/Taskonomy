# Taskonomy
Transfer Taskonomy to noval Tasks with multi-output decoder.<br />
resnet_model.py is the baseline method for apparel classification. It contains a resnet-50 used as encoder.<br />
Load_data.py is to process data<br />
test_all_model can save loss in a CSV file<br />

TO get the representations: python  tools/run_img_task2.py --task class_1000 --input data/dataset/red_dress/names.txt  --output  data/dataset/red_dress/class_1000
