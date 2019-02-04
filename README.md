# App:  Brain Segmentation (patch-based)

Deep learning app made for patch-based T1-weighted MRI brain segmentation using ANTsRNet

## Prerequisites

* [R](https://www.r-project.org) installation
* [ANTsR](https://github.com/ANTsX/ANTsR) installation
* [ANTsRNet](https://github.com/ANTsX/ANTsRNet) installation

## Model training notes

* Training data: IXI, NKI, Kirby, Oasis, ADNI SSTs
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation

## Sample prediction usage

```
#
#  Usage:
#    Rscript doBrainSegmentation.R inputImage inputImageBrainExtractionMask outputImage
#
#  iMac 2011 (with GPU)
#

$ Rscript Scripts/doBrainTissueSegmentation.R Data/Example/1097782_defaced_MPRAGE.nii.gz Data/Example/1097782_defaced_MPRAGEBrainExtractionMask.nii.gz output

*** Successfully loaded .Rprofile ***

Loading required package: ANTsRCore

Attaching package: ‘ANTsRCore’

The following object is masked from ‘package:stats’:

    var

The following objects are masked from ‘package:base’:

    all, any, apply, max, min, prod, range, sum

Using TensorFlow backend.
Loading weights file2019-02-04 15:36:46.214289: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:857] OS X does not support NUMA - returning NUMA node zero
2019-02-04 15:36:46.214466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: TITAN Xp
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:c3:00.0
Total memory: 12.00GiB
Free memory: 10.62GiB
2019-02-04 15:36:46.214484: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2019-02-04 15:36:46.214493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2019-02-04 15:36:46.214512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN Xp, pci bus id: 0000:c3:00.0)
2019-02-04 15:36:46.214882: E tensorflow/stream_executor/cuda/cuda_driver.cc:924] failed to allocate 10.09G (10830902272 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY
  (elapsed time: 1.060265 seconds)
Reading  Data/Example/1097782_defaced_MPRAGE.nii.gz  (elapsed time: 0.8865328 seconds)
Extracting patches based on mask.  (elapsed time: 3.736066 seconds)
968/968 [==============================] - 15s 15ms/step
 (elapsed time: 26.59697 seconds)
Reconstruct from patches and write to disk.  (elapsed time: 36.7814 seconds)

Total elapsed time: 1.133395 seconds

```

## Sample results

![Brain extraction results](Documentation/Images/resultsBrainSegmentation.png)
