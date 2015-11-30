# Face kiosk

Demo app for testing face recognition models. It captures data from webcam, performs face detection and shows 5 most similar faces.

## Installation

### Windows

In Windows install [Anaconda](https://www.continuum.io/downloads) and then follow instructions for installing [precompiled Caffe](http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/). You can use it either with or without CUDA.

### Ubuntu

In Ubuntu first install Caffe:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install the python-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, enable USE_CUDNN or CPU_ONLY)
make all
make test
make runtest
```

Then install required Python packages:
```
sudo apt-get install python-opencv python-numpy python-sklearn
```

Finally clone the repository:
```
git clone https://github.com/tambetm/face_kiosk.git
```

**NB!** While repository uses [Git Large File Storage](https://git-lfs.github.com/) to host large model and index files, it is better to perform clone without git-lfs enabled and download required models manually from links below. By default folders contain placeholders for large files, you can replace them with downloaded files.

## How to run

Before you can run the application, you have to download corresponding images, model and index file. If unsure, start with `CASIA_lfw_oversample.[sh|bat]`, which is of reasonable size, runs fast even on CPU and has decent results.

 * `VGG_lfw.[sh|bat]` - LFW dataset using VGG face model. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/vgg_lfw/vgg_lfw.pkl?raw=true)
 * `VGG_lfw_oversample.sh` - LFW dataset using VGG face model with oversampling. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/vgg_lfw/vgg_lfw_oversample.pkl?raw=true)
 * `CASIA_lfw.[sh|bat]` - LFW dataset using model trained on CASIA-WebFace. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/CASIA_iter_450000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/lfw_100_all.pkl?raw=true)
 * `CASIA_lfw_oversample.[sh|bat]` - LFW dataset using model trained on CASIA-WebFace with oversampling. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/CASIA_iter_450000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/CASIA_lfw_oversample.pkl?raw=true)
 * `CASIA.[sh|bat]` - CASIA-WebFace dataset using model trained on CASIA-WebFace. [(images)](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) [(model)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/CASIA_iter_450000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/CASIA/CASIA.pkl?raw=true)
 * `CASIA_oversample.[sh|bat]` - CASIA-WebFace dataset using model trained on CASIA-WebFace with oversampling. [(images)](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) [(model)](https://github.com/tambetm/face_kiosk/blob/master/CASIA_lfw/CASIA_iter_450000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/CASIA/CASIA_oversample.pkl?raw=true)
 * `lfw.[sh|bat]` - LFW dataset using model trained on LFW extended with WLF. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://github.com/tambetm/face_kiosk/blob/master/lfw/lfw+wlf_iter_130000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/lfw/lfw_all.pkl?raw=true)
 * `fotis.[sh|bat]` - Fotis dataset using model trained on LFW+WLF+Fotis.  [(model)](https://github.com/tambetm/face_kiosk/blob/master/fotis/lfw+wlf+fotis_iter_110000.caffemodel?raw=true)  [(index)](https://github.com/tambetm/face_kiosk/blob/master/fotis/fotis_unlabeled.pkl?raw=true)

Download the files and move them to respective folders, you can overwrite the git-lfs placeholder files. If unsure of the location, check the respective script source. Once this is done, just run the script.

**NB!** VGG face model must be downloaded separately from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz). Unpack the file in application folder, which should result in folder `vgg_face_caffe` containing all the required files. VGG face model will benefit a lot from GPU.

## How to prepare a new dataset

Your dataset should be organized into folders, folder name should be person's name (first name and surname separated by underscore) and it should contain **cropped** face images of this person.

Once you have folder structure in place, you can extract features with following command:

```
python extract.py <images_path> <features.npz> <metadata.csv> <options>
```

First parameter is path to images folder (which has person subfolders), second parameter is name for features file (saved in Numpy .npz format), third parameter is text file name, that will contain file names and person names (saved in CSV format). 

Depending on the model you also need to provide additional options:
 * `--model_file` - Caffe .prototxt file (deploy version)
 * `--pretrained_file` - Caffe .caffemodel file
 * `--mean_file` - Caffe image mean file, either in .binaryproto or .npy format
 * `--image_size` - original image size before cropping, all images are resized to this size
 * `--grayscale` - if input to your network is grayscale image (1 channel)
 * `--oversample` - average features over 10 cropped images (4 corners + center + 2 mirrors of each)
 * `--layer` - name of layer, which features to use
 * `--backend` - either `gpu` or `cpu`
 * `--filter` - which files to look at, e.g. `*.jpg`

See `extract_CASIA_lfw_oversample.sh` for example. Once you have extracted the features in `.npz` file, you need to create index for nearest neighbor search with following command:

```
python build_index.py <features.npz> <index.pkl>
```

Then you need to create a script or batch file for launching the kiosk with proper options. See the example scripts above.
