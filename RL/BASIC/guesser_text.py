import argparse
from fastai.data.load import DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import RL.utils as utils
from sklearn.metrics import confusion_matrix
from transformers import AutoModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory",
                    type=str,
                    default="C:\\Users\\kashann\\PycharmProjects\\choiceMira\\RL",
                    help="Directory for saved models")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--num_epochs",
                    type=int,
                    default=10000,
                    help="number of epochs")
parser.add_argument("--hidden-dim1",
                    type=int,
                    default=32,
                    help="Hidden dimension")
parser.add_argument("--hidden-dim2",
                    type=int,
                    default=64,
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
                    default=100,
                    help="Number of validation trials without improvement")

FLAGS = parser.parse_args(args=[])

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np


class Guesser(nn.Module):
    def __init__(self,
                 hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2):

        super(Guesser, self).__init__()

        self.numeric_X, self.text_features, self.y, self.features_size,_,_ = utils.load_text_data()



        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_embedding_dim = self.text_model.config.hidden_size

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
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, hidden_dim2),
            torch.nn.PReLU(),
        )

        # output layer
        self.logits = nn.Linear(hidden_dim2, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)
        self.path_to_save = os.path.join(os.getcwd(), 'model_guesser_text')

    def embed_text(self, text_batch):
        embeddings_text = []
        for line in text_batch:
            tokens = self.tokenizer(str(line), padding=True, truncation=True, return_tensors='pt', max_length=128)
            with torch.no_grad():
                outputs = self.text_model(**tokens)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming you want to use the [CLS] token
                embeddings_text.append(embeddings)
        # Assuming text_embeddings is your list of tensors
        embeddings_text = torch.stack(embeddings_text, dim=0).squeeze()

        return embeddings_text

    def forward(self, x_numeric, x_text):
        x_combined = torch.cat((x_numeric, x_text), dim=1)
        # Pass through MLP layers
        x_combined = torch.where(torch.isnan(x_combined), torch.zeros_like(x_combined), x_combined)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(x_combined.shape[1], FLAGS.hidden_dim1),
            torch.nn.PReLU(),
        )
        x_combined = self.layer1(x_combined)
        x_combined = self.layer2(x_combined)
        x_combined = self.layer3(x_combined)
        x_combined = self.layer4(x_combined)
        x_combined = self.layer5(x_combined)
        logits = self.logits(x_combined)

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


def val(model, val_loader, best_val_auc=0):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for numeric_data, text_data, labels in val_loader:
            numeric_data = numeric_data.float()
            output = model(numeric_data, text_data)
            _, predicted = torch.max(output.data, 1)
            # Append true and predicted labels to calculate confusion matrix
            y_true.extend(labels.numpy())  # Assuming labels is a numpy array
            y_pred.extend(predicted.numpy())  # Assuming predicted is a numpy array

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
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
        for numeric_data, text_data, labels in test_loader:
            numeric_data = numeric_data.float()
            output = model(numeric_data, text_data)
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


def train_model(model, nepochs, train_loader, val_loader):
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
    accuracy_list = []
    training_loss_list = []

    for epoch in range(1, nepochs + 1):
        running_loss = 0.0
        model.train()

        for numeric_data, text_data, labels in train_loader:
            numeric_data = numeric_data.float()
            model.optimizer.zero_grad()
            output = model(numeric_data, text_data)

            labels = labels.long()
            loss = model.criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            model.optimizer.step()

        training_loss_list.append(running_loss / len(train_loader))

        if epoch % 50 == 0:
            new_best_val_auc = val(model, val_loader, best_val_auc)
            accuracy_list.append(new_best_val_auc)
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1

            if val_trials_without_improvement == FLAGS.val_trials_wo_im:
                print(f'Did not achieve val AUC improvement for {FLAGS.val_trials_wo_im} trials, training is done.')
                break


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

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        model.numeric_X, model.y, model.text_features, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val, text_train, text_val = train_test_split(
        X_train, y_train, text_train, test_size=0.2, random_state=24)

    # Convert numeric data to PyTorch tensors
    X_tensor_train = torch.from_numpy(X_train).float()
    y_tensor_train = torch.from_numpy(y_train).long()
    X_tensor_val = torch.from_numpy(X_val).float()
    y_tensor_val = torch.from_numpy(y_val).long()
    X_tensor_test = torch.from_numpy(X_test).float()
    y_tensor_test = torch.from_numpy(y_test).long()

    text_embeddings = model.embed_text(text_train)
    text_val = model.embed_text(text_val)
    text_test = model.embed_text(text_test)

    # Create DataLoaders for numeric data
    dataset_train = TensorDataset(X_tensor_train, text_embeddings, y_tensor_train)
    dataset_val = TensorDataset(X_tensor_val, text_val, y_tensor_val)
    dataset_test = TensorDataset(X_tensor_test, text_test, y_tensor_test)

    data_loader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=FLAGS.batch_size, shuffle=True)

    train_model(model, FLAGS.num_epochs, data_loader_train, data_loader_val)

    # Tokenize text data
    text_train_tokens = model.tokenizer(list(text_train), padding=True, truncation=True, return_tensors='pt')
    text_val_tokens = model.tokenizer(list(text_val), padding=True, truncation=True, return_tensors='pt')
    text_test_tokens = model.tokenizer(list(text_test), padding=True, truncation=True, return_tensors='pt')

    # Create DataLoaders for numeric and text data
    dataset_train = TensorDataset(X_tensor_train, text_train_tokens['input_ids'],
                                  text_train_tokens['attention_mask'], y_tensor_train)
    dataset_val = TensorDataset(X_tensor_val, text_val_tokens['input_ids'], text_val_tokens['attention_mask'],
                                y_tensor_val)
    dataset_test = TensorDataset(X_tensor_test, text_test_tokens['input_ids'], text_test_tokens['attention_mask'],
                                 y_tensor_test)

    data_loader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=FLAGS.batch_size, shuffle=True)

    train_model(model, model.tokenizer, FLAGS.num_epochs, data_loader_train, data_loader_val)
    test(data_loader_test, model.path_to_save)


if __name__ == "__main__":
    os.chdir(FLAGS.directory)
    main()
