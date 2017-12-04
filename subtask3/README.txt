1. Use "./makemangs" to create magnitude and angle gradient images in the "mangs" folder from the test images in the "tests" folder
2. Use "./makehoughs x y" to create hough images by first thresholding the magnitude images by x and using this with an angle range
    proportional to the width of the image given by y and store the resulting hough images in the "houghs" folder.
3. Use "./makereverse" to create reverse hough images in the "reverse" folder from the hough images in the "houghs" folder.
    (This will take a long time).
4. Use "./detect n" to create two versions of the test images, one with default detected boxes from the cascade classifier in
    the "dartcascade" folder, and the other with some of these detected boxes filtered out using brightness values from the
    reverse hough image. In addition, the file "bounding.txt" will be written to with the co-ordinates of all detected boxes
    from the filtered image. The n parameter is a double that corresponds to the top percentage (compared to the highest value from all boxes)
    of bounding boxes that we are willing to allow through the filter.
5. Use "./bounding bounding.txt" to generate the F-scores and TPR percentages for every test image, with the average F-Score
    for the entire set of images given at the end. This can be compared to using "./bounding default_bounding.txt" which
    outputs the same results but for the images with ALL detected boxes (before they were subseqently filtered in order to
    potentially improve the f-score of the images).
