import json
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import collections
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Model



# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open ("data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open ("data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)



print("Loading the model...")
model = load_model('./model_13.keras')

resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)



# Generate Captions for a random image
# Using Greedy Search Algorithm

def predict_caption(photo, beam_index=3):
    start = [word_to_index["startseq"]]
    sequences = [[start, 0.0]]  # List of [sequence, score]

    # max length of caption
    max_len = 80

    while len(sequences[0][0]) < max_len:
        all_candidates = []

        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_len, padding='post')
            preds = model.predict([photo, padded], verbose=0)[0]
            top_preds = np.argsort(preds)[-beam_index:]  # Top beam_index words

            for word in top_preds:
                next_seq = seq + [word]
                new_score = score + np.log(preds[word] + 1e-10)  # add log prob
                all_candidates.append([next_seq, new_score])

        # sort by score and select best beam_index sequences
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_index]

    # Pick best sequence
    final_seq = sequences[0][0]

    # Convert to words
    final_caption = [index_to_word[i] for i in final_seq]

    # Clean up 'startseq' and 'endseq'
    final_caption = final_caption[1:]
    if "endseq" in final_caption:
        final_caption = final_caption[:final_caption.index("endseq")]
    return ' '.join(final_caption)



def preprocess_image (img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img


# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image (img):
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    # feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def runModel(img_name):
    #img_name = input("enter the image name to generate:\t")

    print("Encoding the image ...")
    photo = encode_image(img_name).reshape((1, 2048))



    print("Running model to generate the caption...")
    caption = predict_caption(photo)

    img_data = plt.imread(img_name)
    plt.imshow(img_data)
    plt.axis("off")

    #plt.show()
    print(caption)
    return caption
