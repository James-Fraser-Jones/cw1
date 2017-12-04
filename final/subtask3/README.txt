To build the project you MUST have the following files and folders:

1. The "dartcascade" folder with the classifier xml files.
2. The folder of test images named "tests".
3. Several empty folders that the programs will use to write files into: "mangs", "houghs", "reverse" and "detections".
4. The following source code files: "makemangs.cpp", "makehoughs.cpp", "makereverse.cpp", "detect.cpp" and "bounding.hs".

Compile the source code files with the following lines:

g++ -O3 makemangs.cpp -o makemangs /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

g++ -O3 makehoughs.cpp -o makehoughs /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

g++ -O3 makereverse.cpp -o makereverse /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

g++ -O3 detect.cpp -o detect /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

ghc -O2 bounding.hs -o bounding

Run the executable files in the same order as given above. Each executable will write files into their corresponding
empty directory. You must wait for the executable to finish and for all the images to be written to the directory
before running the next one. The "makereverse" exectuable in particular will take a long time.

Run each executable using the following lines:

./makemangs
./makehoughs
./makereverse
./detect
./bounding bounding.txt

At the end of this process, you will have all relevent images for all test images at each stages of the classification process. And you will also have the F-Score and TPR% for each test image and the average F-score for all images printed to the screen by the "bounding" executable.
