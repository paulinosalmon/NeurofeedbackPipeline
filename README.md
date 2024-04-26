![image](https://github.com/paulinosalmon/NeurofeedbackPipeline/assets/48361860/12815068-ae21-4acf-94c5-1fea43067418)# NeurofeedbackPipeline

## I. Directory Contents
### The repository consists of 2 major directories:
- attentionPipeline
    - Contains the base and main framework of the [MIT pipeline](https://github.com/gretatuckute/ClosedLoop). This base pipeline contains the code that is describe in [the paper of this study](https://pubmed.ncbi.nlm.nih.gov/33513324/#:~:text=The%20neurofeedback%20code%20framework%20is,for%20scientific%20and%20translational%20use). This code utilizes a closed-loop decoded neurofeedback network to perform mental training that improves the attentional states of those involved. Further description of the specs can be found on the full paper. A simple logistic regression model was used as the decoder of the system. This baseline will be augmented with improved machine learning techniques and transformer technology to add adaptability to its features.
- depressionPipeline
    - The main study's pipeline. Should have the codes for augmented versions of the base one from MIT.
 
## II. Running the system
1. Don't forget to create a virtual environment first by installing it if you haven't (```sudo apt install python3.10-venv```) and creating a venv in the directory (```python3 -m venv name_of_virtual_env```)
2. Begin by installing requirements.txt in the base directory. Activate your virtual environment (```source name_of_virtual_env/bin/activate```) and running ```pip install -r requirements.txt```.
3. The code makes use of two windows running on the matplotlib backend ```TKAgg``` (the xlaunch servers were deprecated). One window simulates the live EEG data gathering window across all channels involved. The other one displays the feedback signal generated by the system. The second window also displays the likelihood probability of the classifier output, as well as some additional logs per script to easier navigate the data flow.
4. System settings can be modified in ```settings.py```. The DecNef can be switched between training state and neurofeedback state. Its functionality will differ accordingly based on this setting.

Major components from the [base framework](https://github.com/gretatuckute/ClosedLoop):
![image](https://github.com/paulinosalmon/NeurofeedbackPipeline/assets/48361860/d6610333-15ff-45a6-aad3-ab533656da24)

## III. Script Information (WIP)
- **settings.py**
- **data_acquisition.py**
- **_1_data_preprocessing.py**
- **_2_artifact_rejection.py**
- **_3_classifier.py**
- **_4_feedback_generator.py**
- **_5_pipeline.py**

