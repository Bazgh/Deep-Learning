# Deep-Learning
Image Captioning
# Deep-Learning
Image Captioning
Task description:
In this Project, we built 5 different models upon eachother to genarate captions for a set of images. We used encoder-decoder architecture and this is the same for all the 5 models. The image is passed into the encoder, and using a pre-trained model, its features are extracted which are later on projected to the embedding space. This ensures all iamges are mapped to a space with the same dimension. The decoder receives the embedded iamge and uses this information along what it knows from the history of the caption (the previous generated words) to generate the next word. This loop starts as soon as the decoder receives the SOS token which is the start of sequence, and stops as the decoder reaches the eos Token which is the end of the sequence. Given this description, being the backbone of our 5 models we try to start from the simplest models and proceed with more complex ones to get the desired result which is generataing correct captions. Our critera for measuring the performance of our models are BLUE Score and loss function convergence illustrated in plots.
# Model_1
Encoder: The pre-trained model is resnet.
Decoder: RNN is used for training.
#Result:
BLUE Score: ![image](https://github.com/user-attachments/assets/02ec99a5-2a1f-440a-b901-fe33e4d60bbd)
Genarated captions for a test image:![image](https://github.com/user-attachments/assets/0a232280-3fe8-4fce-8361-527d0ac76168)
# Model_2
Encoder: The pre_trained model is DINOV which does a great job in feature extraction.
Decoder: RNN is used for training.
BLUE Score: ![image](https://github.com/user-attachments/assets/a819339c-c454-4d0d-b05f-db6d0b3cb926)
Generated Captions for the Test Image:![image](https://github.com/user-attachments/assets/11e2b6c6-c941-4abd-8c39-354b4675bd8e)
# Model_3
Encoder:The pre_trained model is DINOV which does a great job in feature extraction.
Decoder: A Gated network (LSTM) is utilised instead of RNN to capture the long term dependencies.    
BLUE Score: ![image](https://github.com/user-attachments/assets/e735e334-840f-4881-a6ec-c2303ad4d1d8)
Genarated captions for a test image:![image](https://github.com/user-attachments/assets/01d84884-f707-4538-81d9-ce00272b7926)
# Model_4
Encoder:The pre_trained model is DINOV which does a great job in feature extraction.
Decoder: A Gated network (LSTM) is utilised instead of RNN to capture the long term dependencies. 
Note: In this Model, we modified the inputs of the decoder. IN te previous models, the decoder had to distinguish whether the input is an encoded image or the caption indice, and this makes the model prone to mistakes. Thus, we modified this so thata the input of the decoder at eachtime point is only the caption indice. The encoded image this in this model is used as the initial hidden state instead of being backed with the caption incide. 
BLUE Score:![image](https://github.com/user-attachments/assets/3f272b36-a560-4e49-ab6b-66ab4453500f)
Genarated captions for a test image: ![image](https://github.com/user-attachments/assets/580476bc-4b12-4904-a961-8fee5b6f188c)
# Model_6
Encoder:The pre_trained model is DINOV which does a great job in feature extraction. IN the previous models, we used the <CLS> Token extracted by DINOV. However, DINOV has got htis particular feature topextract spatial feartures(Local features) which improves the image captioning task' result. This necessitates the implementation of MultiHEAd attention in both Encoder and Decoder.
BLUE Score: ![image](https://github.com/user-attachments/assets/3125de95-2d88-4ea2-93b4-d8d73584af6c)
Genarated captions for a test image: ![image](https://github.com/user-attachments/assets/8b50722e-022f-433a-961c-b7676de3a760)


