from mmdet.apis import init_detector, inference_detector


fname = "work_dirs/r3det_r50_fpn_2x_20200616/epoch_24.pth"
config_file = 'configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py'
checkpoint_file = 'work_dirs/r3det_r50_fpn_2x_20200616/epoch_24.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
img = 'vis1.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
