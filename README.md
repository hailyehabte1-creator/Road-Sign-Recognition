=>Road Sign Recognition (GTSRB) using Transfer LearningThis project implements a high-accuracy Road Sign Recognition system using Transfer Learning with the MobileNetV2 architecture, 
trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset. 
The objective is to efficiently classify 43 distinct traffic sign classes, prioritizing high accuracy and fast inference suitable for deployment.
 =>Project OverviewThe system utilizes the pre-trained MobileNetV2 model (on ImageNet) as a powerful feature extractor. Training is executed in a phased approach within Google Colab, leveraging GPU resources for accelerated processing.
=> The overall methodology involves data preparation, model construction, two-stage training (frozen feature extraction followed by fine-tuning), and comprehensive evaluation. An interactive Gradio interface is integrated for real-time demonstration and testing.
=>Team Information This project was completed by students from the Debre Berhan University
-College of Computing, 
-Department of Computer Science.
-Course: Selected Topics in Computer Science (CoSc4132)
-NameID  .............    -No
1,Hailye H/Giworgisdbue/0746/13
2,Adamu Abebawdbue/0701/13
3,Tsige Tilahundbue/0788/13
4,Yemisrach Girmadbue/0792/13
5,Leul Aschenakidbue/0754/13
=>Dataset and Model ArchitectureThe system uses the GTSRB (German Traffic Sign Recognition Benchmark) dataset, sourced from Kaggle. The dataset contains 43 unique traffic sign classes.Model ArchitectureThe architecture is built upon the pre-trained MobileNetV2 base, structured for efficient classification:Frozen Convolutional Layers: The MobileNetV2 base is initially frozen during training to use its powerful, pre-learned features.Custom Classification Head: This head is added for final predictions and includes Global Average Pooling, Dropout regularization, and a final Dense softmax classification layer (43 units).Fine-tuning: After initial training, the entire model is unfrozen and trained at a very low learning rate to slightly adjust the weights of the MobileNetV2 base, optimizing it specifically for the road sign domain.
=>Training PipelineThe process follows a structured sequence within the notebook environment:Dependencies and API Setup: Install required Python libraries and configure the Kaggle API (requiring the upload of kaggle.json).Data Preparation: Download, extract, and load images using TensorFlow's image_dataset_from_directory.Splitting and Augmentation: Create Train, Validation, and Test splits and apply essential data augmentation techniques to the training set.Phased Training:Train the model with the MobileNetV2 layers frozen.Fine-tune the entire model at a low learning rate.Evaluation: Evaluate performance on the dedicated test set.Saving and Demo: Save the final model to the gtsrb_mobilenetv2_model/ directory and initiate the Gradio interface.
=> Evaluation Metrics and DemoEvaluation MetricsThe model's performance is rigorously measured using:Overall AccuracyPrecision, Recall, and F1-Score (calculated per class)A Confusion Matrix for detailed performance analysis.DemoA Gradio interface enables interactive testing. Users can upload an image and receive the Predicted sign class and its corresponding Confidence score in real-time.
=>How to Run (Colab)Upload your kaggle.json file to the Colab environment.Execute the notebook cells in sequential order to install dependencies, train the model, and run the Gradio demo.RequirementsPython 3.9+TensorFlow 2.xScikit-learnGradioKaggle API access
