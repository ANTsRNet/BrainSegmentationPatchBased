
################################################################################
Sys.setenv( "CUDA_VISIBLE_DEVICES"=0 )
################################################################################
library(ANTsRNet)
library(ANTsR)
library(keras)
# /home/avants/data/breacher/BIDSProcessed/0102/sub-0102/ses-initial-day2/anat
ddir = "/home/avants/data/breacher/BIDSProcessed/"
segs = Sys.glob( paste0( ddir, "*/*/*/anat/*BrainSegmentation.nii.gz" ) )
n4s = Sys.glob( paste0( ddir, "*/*/*/anat/*BrainSegmentation0N4.nii.gz" ) )
netnm = '~/data/act2014Aff/unetPatchSeg3DdeeperB.h5'
unet = load_model_hdf5( netnm )
template = antsImageRead( "~/code/PBDRKbrain/Data/S_template3_BrainCerebellum.nii.gz" )
for ( k in 1:length(n4s) ) {
  ofn = basename( n4s[k] )
  ofn = tools::file_path_sans_ext( ofn, T )
  odir = "/home/avants/data/breacher/brainAge/"
  if ( ! file.exists(  paste0( odir, ofn, "_seg.nii.gz" ) ) ) {
    cat( paste0("...",k,"*") )
    img = antsImageRead( n4s[k] ) %>% iMath("Normalize")
    msk = thresholdImage( antsImageRead( segs[k] ), 1, 6 )
    img = img * msk
    img = antsRegistration( template, img, 'Affine' )$warpedmovout %>% iMath("Normalize")
    msk = thresholdImage( img, 1e-4, 1 ) %>% iMath("MD",2)
    p = 80
    patchSize = rep( p, 3 )
    strl = patchSize/2
    ptch = extractImagePatches( img, patchSize, maxNumberOfPatches = 'all',
      returnAsArray = TRUE, strideLength = strl, maskImage = msk )
    X = array( ptch, c( dim( ptch ), 1 ) )
    predictedData = predict( unet, X, batch_size = 128 )
    ###############################################
    plist = list()
    for ( k in 1:tail(dim(predictedData),1) ) {
      gmPatch = array( predictedData[,,,,k], dim( ptch ) )
      reconGM = reconstructImageFromPatches( gmPatch, msk,
        domainImageIsMask=FALSE, strideLength = strl )
      plist[[k]] = reconGM
      }
    msk0 = msk
    segvec = apply( imageListToMatrix( plist, msk0 ), FUN=which.max, MARGIN=2 )
    seg = makeImage( msk0, segvec )
    # segimggt = thresholdImage( img, "Otsu", 3 )
    # print( diceOverlap( segimggt, seg ) )
    # plot( img, seg, axis=3, nslices=21, ncolumns=7 )
    # plot( img, plist[[3]], axis=3, nslices=21, ncolumns=7 )
    antsImageWrite( plist[[3]], paste0( odir, ofn, "_wmprob.nii.gz" ) )
    antsImageWrite( plist[[2]], paste0( odir, ofn, "_gmprob.nii.gz" ) )
    antsImageWrite( seg, paste0( odir, ofn, "_seg.nii.gz" ) )
    } else print( paste( ofn , 'done' ) )
  }

