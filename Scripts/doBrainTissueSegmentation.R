library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainTissueSegmentation.R",
    " inputFile inputMaskFile outputFilePrefix\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  inputMaskFileName <- args[2]
  outputFilePrefix <- args [3]
  }

patchSize <- c( 40, 40, 40 )
strideLength <- patchSize / 2

classes <- c( "Background", "Csf", "GrayMatter", "WhiteMatter" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1" )
channelSize <- length( imageMods )

unetModel <- createUnetModel3D( c( patchSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, dropoutRate = 0.0,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/brainSegmentationPatchBasedWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "brainSegmentationPatchBased", weightsFileName )
  }
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 ) %>% iMath( "Normalize" )
mask <- antsImageRead( inputMaskFileName, dimension = 3 )
mask[mask != 0] <- 1
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Extracting patches based on mask." )
startTime <- Sys.time()

imagePatches <- extractImagePatches( image, strideLength = strideLength,
  patchSize, maxNumberOfPatches = 'all', maskImage = mask,
  returnAsArray = TRUE )
batchX <- array( data = 0, dim = c( dim( imagePatches )[1], patchSize, 1 ) )
batchX[,,,,1] <- imagePatches

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedDataArray <- unetModel %>% predict( batchX, verbose = 1 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Reconstruct from patches" )
startTime <- Sys.time()

cleanedImage <- reconstructImageFromPatches( drop( predictedDataArray ),
  mask, strideLength = strideLength, domainImageIsMask = FALSE )

antsImageWrite( cleanedImage, outputFileName )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )







cat( "Normalizing to template and cropping to mask." )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
warpedMask <- applyAntsrTransformToImage( xfrm, mask, reorientTemplate,
  interpolation = "nearestNeighbor" )
warpedMask <- iMath( warpedMask, "MD", 3 )
warpedCroppedImage <- cropImage( warpedImage, warpedMask, 1 )
originalCroppedSize <- dim( warpedCroppedImage )
warpedCroppedImage <- resampleImage( warpedCroppedImage,
  resampledImageSize, useVoxels = TRUE )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

batchX <- array( data = as.array( warpedCroppedImage ),
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, warpedCroppedImage )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Renormalize to native space" )
startTime <- Sys.time()

zeroArray <- array( data = 0, dim = dim( warpedImage ) )
zeroImage <- as.antsImage( zeroArray, reference = warpedImage )

probabilityImages <- list()
for( i in seq_len( numberOfClassificationLabels ) )
  {
  probabilityImageTmp <- resampleImage( probabilityImagesArray[[1]][[i]],
    originalCroppedSize, useVoxels = TRUE )
  probabilityImageTmp <- decropImage( probabilityImageTmp, zeroImage )
  probabilityImages[[i]] <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImageTmp, image )
  }

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Writing", outputFilePrefix )
startTime <- Sys.time()

probabilityImageFiles <- c()
for( i in seq_len( numberOfClassificationLabels - 1 ) )
  {
  probabilityImageFiles[i] <- paste0( outputFilePrefix, classes[i+1], ".nii.gz" )
  antsImageWrite( probabilityImages[[i+1]], probabilityImageFiles[i] )
  }

probabilityImagesMatrix <- imagesToMatrix( probabilityImageFiles, mask )
segmentationVector <- apply( probabilityImagesMatrix, FUN = which.max, MARGIN = 2 )
segmentationImage <- makeImage( mask, segmentationVector )
antsImageWrite( segmentationImage, paste0( outputFilePrefix, "Segmentation.nii.gz" ) )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
