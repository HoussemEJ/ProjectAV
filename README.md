# Project: Benchmarking 2D state-of-the-art human pose detectors using egocentric perspective

This is a quick guide on how to setup the enviroment.

### Requirements:
* Ubuntu 20.04
* Cuda 11.7
* Pytorch 1.9.0
### Setup
* First clone the vitpose repo and follow installation instructions https://github.com/ViTAE-Transformer/ViTPose
* Copy the "bottom_up_img_demo.py" and "top_down_img_demo_with_mmdet" into https://github.com/ViTAE-Transformer/ViTPose/tree/main/demo
* You can optionally run FramesToVideo.py and VideoToFrames.py to convert your data.
* Run the scripts "hrformer_pred.py" "vitpose_l_pred.py" "vitpose_s_pred.py" you can also check the "Commands.txt file for details on how to run the detectors.
* After getting the predictions run "pycocotools_results.py" and optionally "compute_ap.py", "compute_OKS.py" to get your evaluation results.
