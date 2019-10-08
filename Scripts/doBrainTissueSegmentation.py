#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et

import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

args = sys.argv

if len(args) != 4:
    help_message = ("Usage:  python doBrainTissueSegmentation.py" +
        " inputFile inputMaskFile outputFilePrefix")
    raise AttributeError(help_message)
else:
    input_file_name = args[1]
    input_mask_file_name = args[2]
    output_file_name_prefix = args[3]

patch_size = (40, 40, 40)
stride_length = tuple(int(t/2) for t in patch_size)

classes = ("Csf", "GrayMatter", "WhiteMatter", "Background")
number_of_classification_labels = len(classes)

image_mods = ["T1"]
channel_size = len(image_mods)

print("Create u-net model.")
unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
  number_of_outputs = number_of_classification_labels,
  number_of_layers = 4,
  number_of_filters_at_base_layer = 16,
  dropout_rate = 0.0,
  convolution_kernel_size = (3, 3, 3),
  deconvolution_kernel_size = (2, 2, 2),
  weight_decay = 1e-5 )

print( "Loading weights file" )
start_time = time.time()
weights_file_name = "./brainSegmentationPatchBasedWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("brainSegmentationPatchBased", weights_file_name)

unet_model.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

# Process input

start_time_total = time.time()

print( "Reading ", input_file_name )
start_time = time.time()
image = ants.image_read(input_file_name)
mask = ants.image_read( input_mask_file_name)
mask = ants.threshold_image( mask, 0.4999, 1.0001, 1, 0)
image = image * mask
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Extracting patches based on mask.")
start_time = time.time()

image_patches = antspynet.extract_image_patches(image, stride_length=stride_length,
  patch_size=patch_size, max_number_of_patches="all", mask_image=mask,
  return_as_array=True)
image_patches = (image_patches - image_patches.min())/(image_patches.max() - image_patches.min())
batchX = np.expand_dims(image_patches, axis=-1)

end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Prediction and decoding")
start_time = time.time()
predicted_data = unet_model.predict(batchX, verbose=1)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Reconstruct from patches and write to disk.")
start_time = time.time()

probability_images_array = list()
for i in range(number_of_classification_labels-1):
    probability_image = antspynet.reconstruct_image_from_patches(
      np.squeeze(predicted_data[:, :, :, :, i]),
      domain_image=mask, stride_length=stride_length,
      domain_image_is_mask=True)
    probability_image=ants.iMath_normalize(probability_image)
    probability_images_array.append(probability_image)

end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

for i in range(number_of_classification_labels - 1):
    print("Writing", classes[i])
    start_time = time.time()
    ants.image_write(probability_images_array[i],
      output_file_name_prefix + classes[i] + "Segmentation.nii.gz")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")

probability_images_matrix = ants.image_list_to_matrix(probability_images_array,mask)
segmentation_vector = np.argmax(probability_images_matrix, axis=0) + 1 
segmentation_image = ants.make_image(mask, segmentation_vector)
ants.image_write(segmentation_image, output_file_name_prefix + "Segmentation.nii.gz")

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total
print("Total elapsed time: ", elapsed_time_total, "seconds")
