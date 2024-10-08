import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
import argparse
import numpy as np
import RL.utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default=r'C:\Users\kashann\PycharmProjects\PCAFE\RL\Integration',
                    help="Directory for saved models")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=1000,
                    help="number of epochs")
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=64,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=128,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.001,
                    help="l_2 weight penalty")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=20,
                    help="Number of validation trials without improvement")

FLAGS = parser.parse_args(args=[])


class ImageEmbedder(nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel images (MNIST)
        in_channels = self.resnet.conv1.in_channels
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Allow fine-tuning of all layers
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Remove the classification layer to get embeddings
        self.resnet.fc = torch.nn.Identity()

        # Add a fully connected layer to map the 2048-dim output to 768-dim
        self.fc = nn.Linear(2048, 768)

        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize MNIST images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    def embed_image(self, image_path):
        image = Image.open(image_path).convert('L')  # Ensure the image is in grayscale
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Generate embedding
        embedding = self.resnet(img_tensor)  # Extract features using ResNet
        embedding = self.fc(embedding)  # Reduce to 768-dim

        return embedding


def map_features_to_indices(data):
    index_map = {}
    current_index = 0

    for i, feature in enumerate(data):
        if isinstance(feature, str):  # Check if feature is text
            index_map[i] = list(range(current_index, current_index + 10))
            current_index += 10
        else:  # Assume it's numeric if not text
            index_map[i] = [current_index]
            current_index += 1

    return index_map


class MultimodalGuesser(nn.Module):
    def __init__(self, hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2, text_embed_dim=768, img_embed_dim=768, text_reduced_dim=30, img_reduced_dim=30):
        super(MultimodalGuesser, self).__init__()
        self.X, self.y, self.features_size = RL.utils.load_data_sheba()

        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.img_embedder = ImageEmbedder()
        self.text_reducer = nn.Linear(text_embed_dim, text_reduced_dim)
        self.img_reducer = nn.Linear(img_embed_dim, img_reduced_dim)
        self.text_reduced_dim = text_reduced_dim
        self.num_classes = len(np.unique(self.y))

        # check how many numeric features we have in the dataset
        numeric_features = []
        for i in range(self.features_size):
            if isinstance(self.X.iloc[0, i], (int, float)):
                numeric_features.append(i)
        self.features_total = len(numeric_features) + (self.features_size - len(numeric_features)) * text_reduced_dim
        self.map = map_features_to_indices(self.X.iloc[0])

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.features_total, hidden_dim1),
            torch.nn.PReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.PReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        # output layer
        self.logits = nn.Linear(hidden_dim2, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr, )
        self.path_to_save = os.path.join(os.getcwd(), 'model_robust_embedder_guesser')

    def embed_text(self, text):
        tokens = self.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return F.relu(self.text_reducer(embeddings))

    def embed_image(self, image_path):
        # Get image embedding using ImageEmbedder
        embedding = self.img_embedder.embed_image(image_path)
        # Reduce dimensionality and apply ReLU activation
        return F.relu(self.img_reducer(embedding))



    def is_numeric_value(self, value):
        # Check if the value is an integer, a floating-point number, or a tensor of type float or double
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float, torch.float64]:
                return True
        return False

    def is_text_value(self, value):
        # Check if the value is a string
        if isinstance(value, str):
            return True
        else:
            return False

    def is_image_value(self, value):
        # check if value is path that ends with 'png' or 'jpg'
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False

    def forward(self, input):

        sample_embeddings = []

        for feature in input:
            if self.is_image_value(feature):
                # Handle image path: assume feature is a path and process it
                feature_embed = self.embed_image(feature)
            elif self.is_text_value(feature):
                # Handle text: assume feature is text and process it
                feature_embed = self.embed_text(feature)
            elif self.is_numeric_value(feature):
                # Handle numeric: directly convert to tensor
                feature_embed = torch.tensor([feature], dtype=torch.float32).unsqueeze(0)
            sample_embeddings.append(feature_embed)

        x = torch.cat(sample_embeddings, dim=1)
        x = x.squeeze(dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.logits(x)
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs


def mask(input: np.array, model) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''
    # reduce the input to 1 dim
    input = input.flatten()

    # check if images has 1 dim
    if len(input.shape) == 1:
        masked_input = []
        for i in range(input.shape[0]):


            fraction = 0.2
            if np.random.rand() < fraction:
                if model.is_numeric_value(input[i]):
                    masked_input.append(0)
                else:
                    # append model.text_reduced_dim zeros to the masked_input
                    masked_input.extend([0] * model.text_reduced_dim)
            else:
                masked_input.append(input[i])

        return masked_input


def mask_specific_feature(sample, max_gradients_index, pretrained_model):
    masked_sample = []
    for i, feature in enumerate(sample):
        if i == max_gradients_index:
            if pretrained_model.is_numeric_value(feature):
                masked_sample.append(0)
            else:
                masked_sample.extend([0] * pretrained_model.text_reduced_dim)
        else:
            masked_sample.append(feature)
    return masked_sample


def create_adverserial_input(sample, label, pretrained_model):
    sample_embeddings = []
    for feature in sample:
        if pretrained_model.is_image_value(feature):
            # Handle image path: assume feature is a path and process it
            feature_embed = pretrained_model.embed_image(feature)
        elif pretrained_model.is_text_value(feature):
            # Handle text: assume feature is text and process it
            feature_embed = pretrained_model.embed_text(feature)

        elif pretrained_model.is_numeric_value(feature):
            # Handle numeric: directly convert to tensor
            feature_embed = torch.tensor([feature], dtype=torch.float32).unsqueeze(0)
        sample_embeddings.append(feature_embed)

    input = torch.cat(sample_embeddings, dim=1)
    input = input.squeeze(dim=1)
    pretrained_model.eval()

    # Set requires_grad to True to calculate gradients with respect to input
    input = input.detach().clone().requires_grad_(True)
    # Forward pass
    x = pretrained_model.layer1(input)
    x = pretrained_model.layer2(x)
    x = pretrained_model.layer3(x)

    logits = pretrained_model.logits(x)
    if logits.dim() == 2:
        probs = F.softmax(logits, dim=1)
    else:
        probs = F.softmax(logits, dim=-1)
    # Calculate loss
    loss = pretrained_model.criterion(probs, label)

    # Backward pass to get gradients of the loss with respect to the input
    loss.backward()
    # Get the absolute value of the gradients
    gradient = input.grad

    # Identify the most influential features (those with the largest absolute gradients).
    absolute_gradients = torch.abs(gradient)
    max_gradients_index = torch.argmax(absolute_gradients, dim=-1).item()
    for key, value in pretrained_model.map.items():
        if max_gradients_index in value:
            max_gradients_index = key
            break
    return mask_specific_feature(sample, max_gradients_index, pretrained_model)


def train_model(model,
                nepochs, X_train, y_train, X_val, y_val):
    '''
    Train a pytorch model and evaluate it every 2 epoch.
    Params:
    model - a pytorch model to train
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    '''
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    val_trials_without_improvement = 0
    best_val_auc = 0
    # count total sampels in trainloaser
    accuracy_list = []
    num_samples = len(X_train)
    for j in range(1, nepochs):
        running_loss = 0
        random_indices = np.random.choice(num_samples, size=64, replace=False)
        for i in random_indices:

            input = X_train[i]
            label = torch.tensor([y_train[i]], dtype=torch.long)
            if 100 < j < 200:
                input = create_adverserial_input(input, label, model)
            else:
                input = mask(input, model)
            model.train()
            model.optimizer.zero_grad()
            output = model(input)
            loss = model.criterion(output, label)
            running_loss += loss.item()
            loss.backward()
            model.optimizer.step()

        if j % 30 == 0:
            new_best_val_auc = val(model, X_val, y_val, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            if val_trials_without_improvement == FLAGS.val_trials_wo_im:
                print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
                    FLAGS.val_trials_wo_im))


def save_model(model):
    '''
    Save the model to a given path
    :param model: model to save
    :param path: path to save the model to
    :return: None
    '''
    path = model.path_to_save
    if not os.path.exists(path):
        os.makedirs(path)
    guesser_filename = 'best_guesser.pth'
    guesser_save_path = os.path.join(path, guesser_filename)
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(model.cpu().state_dict(), guesser_save_path + '~')
    os.rename(guesser_save_path + '~', guesser_save_path)


def val(model, X_val, y_val, best_val_auc=0):
    correct = 0
    model.eval()
    num_samples = len(X_val)
    with torch.no_grad():
        for i in range(1, num_samples):
            input = X_val[i]
            label = y_val[i]
            input = mask(input, model)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1

    accuracy = correct / num_samples
    print(f'Validation Accuracy: {accuracy:.2f}')
    if accuracy >= best_val_auc:
        save_model(model)
    return accuracy


def test(X_test, y_test, path_to_save):
    X_test = X_test.to_numpy()
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(path_to_save, guesser_filename)
    model = MultimodalGuesser()
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i in range(len(X_test)):
            input = X_test[i]
            label = y_test[i]
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                correct += 1


            y_true.append(label)  # Assuming labels is a numpy array
            y_pred.append(predicted.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Test Accuracy: {accuracy:.2f}')





def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    model = MultimodalGuesser()

    X_train, X_test, y_train, y_test = train_test_split(model.X,
                                                        model.y,
                                                        test_size=0.01,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.05,
                                                      random_state=24)

    train_model(model, FLAGS.num_epochs,
                X_train, y_train, X_val, y_val)

    test(X_test, y_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
