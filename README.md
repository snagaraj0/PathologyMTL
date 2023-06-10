# cs231nFinalProject

Whole Slide Image (WSI) analysis is a common procedure in pathology to identify cancer metastases and grade the trajectory of the cancer. Recently, various deep learning approaches have shown promise in aiding digital WSI analysis by extracting biologically relevant features that can automatically grade slides and segment cancer masses. However, individual supervised models are generally trained and evaluated on a single task with a hyper-specific dataset which limits generalizability and interpretability of these models. In this work, we provide a framework for applying multitask learning to two popular tasks in digital pathology, WSI grading (classification) and segmentation. We select 4,752 WSI images from which we create and preprocess 171,072 256x256 image patches and their associated segmentation masks. Our general multitask learning (MTL) framework involves taking image patches and passing them through a fine-tuned encoder, creating representations that are then bridged to task-specific decoders. Based on this framework, we built four encoding methods: two based on convolutional neural networks, one based on a hierarchical vision transformer and one based on a graph transformer alongside standard task-specific decoding methods. Our results show that our MTL models are able to nearly match individual task-specific baselines, while requiring significantly less training time. Furthermore, visualization of segmentation predictions and classification saliency maps show that models built with our framework are able to extract key biological features. Our findings demonstrate this framework as a promising method to create biologically meaningful WSI representations. 
