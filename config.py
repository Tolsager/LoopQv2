import torch

# CSV_TRAIN = r'challengeA_data/challengeA_data/challengeA_train.csv'
CSV_TRAIN = r'C:\Users\victo\Documents\LoopQ_challenge_A\challengeA_data\challengeA_data\challengeA_train.csv'
# CSV_TEST = r'challengeA_data/challengeA_data/challengeA_test.csv'
CSV_TEST = r'C:\Users\victo\Documents\LoopQ_challenge_A\challengeA_data\challengeA_data\challengeA_test.csv'

# IMAGE_DIRECTORY_TRAIN = r'challengeA_data/challengeA_data/images_train'
IMAGE_DIRECTORY_TRAIN = r'C:\Users\victo\Documents\LoopQ_challenge_A\challengeA_data\challengeA_data\images_train'
# IMAGE_DIRECTORY_TEST = r'challengeA_data/challengeA_data/images_test'
IMAGE_DIRECTORY_TEST = r'C:\Users\victo\Documents\LoopQ_challenge_A\challengeA_data\challengeA_data\images_test'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 24
