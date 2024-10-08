import argparse
import numpy as np
from fastai.data.load import DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import RL.utils as utils
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default="RL/EHR-FS",
                    help="Directory for saved models")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=100,
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
                    default=15,
                    help="Number of validation trials without improvement")

FLAGS = parser.parse_args(args=[])


def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.

    X_noisy = X.copy()
    X_noisy= pd.DataFrame(X_noisy)
    for col in X_noisy.columns:
        categories = X_noisy[col].cat.categories
        noise = np.random.randint(-1, 2, size=len(X))
        X_noisy[col] = (X_noisy[col].cat.codes + noise) % len(categories)
        X_noisy[col] = X_noisy[col].astype('category').cat.set_categories(categories)
    return X_noisy
    """


def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy


def balance_class_multi(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    # Calculate the difference in sample counts for each class
    count_diff = max_class_count - class_counts

    # Initialize arrays to store balanced data
    X_balanced = X.copy()
    y_balanced = y.copy()

    # Add noise to the features of the minority classes to balance the dataset
    for minority_class, diff in zip(unique_classes, count_diff):
        if diff > 0:
            # Get indices of samples belonging to the current minority class
            minority_indices = np.where(y == minority_class)[0]

            # Randomly sample indices from the minority class to add noise
            noisy_indices = np.random.choice(minority_indices, diff, replace=True)

            # Add noise to the features of the selected samples
            X_balanced = np.concatenate([X_balanced, add_noise(X[noisy_indices], noise_std)], axis=0)
            y_balanced = np.concatenate([y_balanced, y[noisy_indices]], axis=0)

    return X_balanced, y_balanced


def balance_class(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, add_noise(X[noisy_indices], noise_std)], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced




class ImageEmbedder(nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Identity()  # Remove the classification layer to get embeddings

        # Add a fully connected layer to map the 2048-dim output to 768-dim
        self.fc = nn.Linear(2048, 768)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Generate embedding
        with torch.no_grad():
            embedding = self.resnet(img_tensor)
            embedding = self.fc(embedding)  # Reduce to 768-dim

        return embedding





class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """

    def __init__(self,
                 hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2):

        super(Guesser, self).__init__()
        self.X, self.y, self.question_names, self.features_size = utils.load_diabetes()
        self.X, self.y = balance_class(self.X, self.y)
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = ImageEmbedder()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.features_size, hidden_dim1),
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
        self.logits = nn.Linear(hidden_dim2, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr, )
        self.path_to_save = os.path.join(os.getcwd(), 'model_robust_guesser')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.logits(x)
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))





def create_adverserial_input(inputs, labels, pretrained_model):
    input = inputs.view(inputs.shape[0], -1).float()

    # Set requires_grad to True for the input tensor
    input.requires_grad = True

    # Forward pass
    output = pretrained_model(input)
    labels = torch.Tensor(labels).long()
    loss = pretrained_model.criterion(output, labels)

    # L1 regularization term
    # l1_reg = torch.tensor(0., requires_grad=True)
    # for param in pretrained_model.parameters():
    #     l1_reg = l1_reg + torch.norm(param, 1)
    #
    # # Add regularization term to the loss
    # loss = loss + 0.001 * l1_reg

    # Compute gradients with respect to input
    loss.backward(retain_graph=True)

    # Get the gradients with respect to the input
    gradient = input.grad

    # Identify the most influential features (those with the largest absolute gradients).
    absolute_gradients = torch.abs(gradient)
    max_gradients_indices = torch.argmax(absolute_gradients, dim=-1)

    # Zero out the most influential features.
    mask = torch.nn.functional.one_hot(max_gradients_indices, num_classes=input.shape[-1]).float()
    zeroed_input_features = input * (1 - mask)

    return zeroed_input_features


def mask(input: np.array) -> np.array:
    '''
    Mask feature of the input
    :param images: input
    :return: masked input
    '''

    # check if images has 1 dim
    if len(input.shape) == 1:
        for i in range(input.shape[0]):
            # choose a random number between 0 and 1
            # fraction = np.random.uniform(0, 1)
            fraction = 0.2
            if (np.random.rand() < fraction):
                input[i] = 0
        return input
    else:
        for j in range(int(len(input))):
            for i in range(input[0].shape[0]):
                # fraction = np.random.uniform(0, 1)
                fraction = 0.2
                if (np.random.rand() < fraction):
                    input[j][i] = 0
        return input

def val(model, val_loader, best_val_auc=0):
    correct = 0
    total = 0
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for input, labels in val_loader:
            input = mask(input)
            input = input.view(input.shape[0], -1)
            input = input.float()
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.criterion(output, labels)
            valid_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.2f}')
    if accuracy >= best_val_auc:
        save_model(model)
    return accuracy


def test(test_loader, path_to_save):
    guesser_filename = 'best_guesser.pth'
    guesser_load_path = os.path.join(path_to_save, guesser_filename)
    model = Guesser()  # Assuming Guesser is your model class
    guesser_state_dict = torch.load(guesser_load_path)
    model.load_state_dict(guesser_state_dict)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = mask(images)
            images = images.view(images.shape[0], -1)
            images = images.float()
            output = model(images)
            _, predicted = torch.max(output.data, 1)

            # Append true and predicted labels to calculate confusion matrix
            y_true.extend(labels.numpy())  # Assuming labels is a numpy array
            y_pred.extend(predicted.numpy())  # Assuming predicted is a numpy array

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f'Test Accuracy: {accuracy:.2f}')


def embed_input(input, model):
    samples = []
    for sample in input:
        input_embedded = []
        for feature in sample:
            if is_image_value(feature):
                feature = embed_image(feature, model)
            elif is_text_value(feature):
                feature = embed_text(feature, model)
            input_embedded.append(feature)
        samples.append(input_embedded)
    # return torch.stack([torch.stack(sample) for sample in samples])



def train_model(model,
                nepochs, train_loader, val_loader):
    '''
    Train a pytorch model and evaluate it every 2 epoch.
    Params:
    model - a pytorch model to train
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    '''
    val_trials_without_improvement = 0
    best_val_auc = 0
    # count total sampels in trainloaser
    accuracy_list = []
    training_loss_list = []
    for i in range(1, nepochs):
        running_loss = 0
        for input, labels in train_loader:
            #create embedding of the input
            input= embed_input(input,model)


            # send images and labels and model to adversarial training
            if (i > 500 and i < 600):
                input = create_adverserial_input(input, labels, model)
            else:
                input = mask(input)

            input = input.view(input.shape[0], -1).float()
            model.train()
            model.optimizer.zero_grad()
            output = model(input)
            y_pred = []
            for y in labels:
                y = torch.Tensor(y).long()
                y_pred.append(y)
            labels = torch.Tensor(np.array(y_pred)).long()
            loss = model.criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            model.optimizer.step()

        training_loss_list.append(running_loss / len(train_loader))
        print(f'Epoch: {i}, training loss: {running_loss / len(train_loader):.2f}')
        if i % 20 == 0:
            new_best_val_auc = val(model, val_loader, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
            # check whether to stop training
            # if val_trials_without_improvement == FLAGS.val_trials_wo_im:
            #     print('Did not achieve val AUC improvement for {} trials, training is done.'.format(
            #         FLAGS.val_trials_wo_im))

    save_plot_acuuracy_epoch(accuracy_list, training_loss_list)


def embed_text(text,model):
    tokens = model.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model.text_model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming you want to use the [CLS] token
    return embeddings


def embed_image(image_path,model):
    embedding = model.embedder.embed_image(image_path)
    return embedding


def is_numeric_value(value):
    # Check if the value is an integer or a floating-point number
    if isinstance(value, (int, float)):
        return True
    else:
        return False


def is_text_value(value):
    # Check if the value is a string
    if isinstance(value, str):
        return True
    else:
        return False


def is_image_value(value):
    # check if value is path that ends with 'png' or 'jpg'
    if isinstance(value, str):
        if value.endswith('png') or value.endswith('jpg'):
            return True
        else:
            return False


def embed_features(input,model):
    # check type of feature

    if is_image_value(input):
        answer_vec = embed_image(input,model)
    elif is_text_value(input):
        answer_vec = embed_text([input],model)
    elif is_numeric_value(input):
        answer_vec = input


def save_plot_acuuracy_epoch(val_accuracy_list, training_loss_list):
    epochs_val = range(1, len(val_accuracy_list) + 1)

    plt.figure(figsize=(10, 5))

    # Plot Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_val, val_accuracy_list, marker='o', linestyle='-', color='b')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    epochs_train = range(1, len(training_loss_list) + 1)
    # Plot Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_train, training_loss_list, marker='o', linestyle='-', color='r')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_loss_plot.png')
    plt.show()


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


def main():
    '''
    Train a neural network to guess the correct answer
    :return:
    '''
    model = Guesser()

    X_train, X_test, y_train, y_test = train_test_split(model.X,
                                                        model.y,
                                                        test_size=0.1,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=24)
    # Convert data to PyTorch tensors
    X_tensor_train = torch.from_numpy(X_train)

    y_tensor_train = torch.from_numpy(y_train)  # Assuming y_data contains integers
    # Create a TensorDataset
    dataset_train = TensorDataset(X_tensor_train, y_tensor_train)

    # Create a DataLoader
    data_loader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size, shuffle=True)
    # Convert data to PyTorch tensors
    X_tensor_val = torch.Tensor(X_val)
    y_tensor_val = torch.Tensor(y_val)  # Assuming y_data contains integers
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    data_loader_val = DataLoader(dataset_val, batch_size=FLAGS.batch_size, shuffle=True)
    train_model(model, FLAGS.num_epochs,
                data_loader_train, data_loader_val)
    X_tensor_test = torch.Tensor(X_test)
    y_tensor_test = torch.Tensor(y_test)  # Assuming y_data contains integers
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
    data_loader_test = DataLoader(dataset_test, batch_size=FLAGS.batch_size, shuffle=True)

    test(data_loader_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
