# MMF-Net-alpha-Matte-Boundary-Defocus-Model-Fusion
Code for our papaer An α-Matte Boundary Defocus Model Based Cascaded Network for Multi-focus Image Fusion

Overview
----
Capturing an all-in-focus image with a single camera is difficult since the depth of field of the camera is usually limited. An alternative method to obtain the all-in-focus image is to fuse several images focusing at different depths. However, existing multi-focus image fusion methods cannot obtain clear results for areas near the focused/defocused boundary (FDB). In this paper, a novel α-matte boundary defocus model is proposed to generate realistic training data with the defocus spread effect precisely modeled, especially for areas near the FDB. Based on this α-matte defocus model and the generated data, a cascaded boundary aware convolutional network termed MMF-Net is proposed and trained, aiming to achieve clearer fusion results around the FDB. More specifically, the MMF-Net consists of two cascaded sub-nets for initial fusion and boundary fusion, respectively; these two sub-nets are designed to first obtain a guidance map of FDB and then refine the fusion near the FDB. Experiments demonstrate that with the help of the new α-matte boundary defocus model, the proposed MMF-Net outperforms the state-of-the-art methods both qualitatively and quantitatively.

![image1](./Illustration/1.png)

α-Matte Boundary Defocus Model
----

![image5](./Illustration/5.png)

![image6](./Illustration/6.png)

Network Architecture
----

![image3](./Illustration/3.png)

Dataset
----

![image4](./Illustration/4.png)

Training
----
run main.py


Testing
----
run run_inference.py

Pre-trained models
----


Results
----
You are very welcome to conduct results comparison with our method. The fusion results on Lytro dataset are shown in ./ResultsLytro.

Acknowledgement
----
Our model is trained on generated dataset, and some of the original images come from Adobe Alpha Matting and Microsoft COCO datasets.

Citation
----
