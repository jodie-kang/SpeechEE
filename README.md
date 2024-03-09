# Towards Event Extraction from Speech with Contextual Clues

- An implementation for Towards Event Extraction from Speech with Contextual Clues.

[comment]: <> (## Update)

[comment]: <> (- **[2023-08-17]** Upload code and data.)

[comment]: <> (## Quick Links)

### Requirements
To set up the required environment, run the following command:
```cmd
conda env create -f environment.yaml
```

### File Structure

```markdown
├─Audio
│  ├─Human-MAVEN
│  ├─Speech-ACE05
│  ├─Speech-DuEE
│  └─Speech-MAVEN
├─code
│  ├─configs
│  ├─data_modules
│  ├─extraction
│  ├─logs
│  ├─models
│  └─output
├─data
│  ├─ACE05
│  ├─ACE05_ASR_W2V2
│  ├─ACE05_ASR_whisper_medium
│  ├─ACE05_clues
│  ├─DuEE
│  ├─DuEE_ASR
│  ├─DuEE_clues
│  ├─Human_MAVEN
│  ├─MAVEN
│  ├─MAVEN_ASR
│  └─MAVEN_clues

```

### Model Training

To train the model, run the following command:
```cmd
python run_main.py fit --config configs/Whisper_medium_ace05_mapper_linear.yaml > output.txt
```

###  Evaluation
To evaluate the trained model, run the following command:
```cmd
python run_main.py test --config configs/Whisper_medium_ace05_mapper_linear.yaml > output_test.txt
```
