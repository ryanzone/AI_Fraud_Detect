MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128        # reduce from 256 (faster)
BATCH_SIZE = 8          # smaller batch for CPU
EPOCHS = 2              # start with 2
LEARNING_RATE = 2e-5