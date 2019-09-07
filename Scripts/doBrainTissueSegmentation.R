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
  outputFilePrefix <- args[3]
  }

patchSize <- c( 40, 40, 40 )
strideLength <- patchSize / 2

classes <- c( "Csf", "GrayMatter", "WhiteMatter", "Background" )
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
unetModel$load_weights( weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 ) %>% iMath( "Normalize" )
mask <- antsImageRead( inputMaskFileName, dimension = 3 )
mask <- thresholdImage( mask, 0.4999, 1.0001, 1, 0 )
image <- image * mask
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

cat( "Reconstruct from patches and write to disk." )
startTime <- Sys.time()

probabilityImageFiles <- c()
for( i in seq_len( numberOfClassificationLabels - 1 ) )
  {
  probabilityImage <- reconstructImageFromPatches( predictedDataArray[,,,,i],
    mask, strideLength = strideLength, domainImageIsMask = TRUE )
  probabilityImageFiles[i] <- paste0( outputFilePrefix, classes[i], ".nii.gz" )
  antsImageWrite( probabilityImage, probabilityImageFiles[i] )
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
