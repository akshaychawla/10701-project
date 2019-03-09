----- Reuse results database ------
1. Download the features database.mat and dictionary vocab.mat
Link download: https://drive.google.com/open?id=1P5vW8p1x5W4Bhs9knQkoyx2Ko_QXbtUc
2. Put dictionary to dir: dictionary/
3. Put features to dir : features/Caltech101

----- Create from scratch ------
VLFear setup matlab:
>> run('vlfeat-0.9.21/toolbox/vl_setup')

1. Put Caltech101 images to images folder with dir: images/Caltech101
2. Extract dense-sift: matlab run:
    >> CalculateDenseSiftDescriptor('images/Caltech101', 'data/Caltech101')
3. Build dictionary vocab.mat file and features database.mat file: matlab run
    >> explicit_kernel_map

----- Classification ----- 
Install sckit learn: pip install scikit-learn
Install h5py: pip install h5py 
Run python dense_sift_classify.py
Draw results mentioned in report: python draw_graph.py
