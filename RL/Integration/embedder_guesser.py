import os

import torch.nn.functional as F
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





class MultimodalNN(nn.Module):
    def __init__(self, hidden_dim1=FLAGS.hidden_dim1, hidden_dim2=FLAGS.hidden_dim2,
                 num_classes=2,text_embed_dim=768, img_embed_dim=2048, text_reduced_dim=10, img_reduced_dim=10):
        super(MultimodalNN, self).__init__()

        # Pre-trained text model (BERT)
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.img_embedder = ImageEmbedder()
        # Separate dimensionality reduction layers
        self.text_reducer = nn.Linear(text_embed_dim, text_reduced_dim)
        self.img_reducer = nn.Linear(img_embed_dim, img_reduced_dim)

        self.features_size = text_reduced_dim + img_reduced_dim
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


    def embed_text(self, text):
        tokens = self.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming you want to use the [CLS] token
        return embeddings

    def embed_image(self, image_path):
        embedding = self.img_embedder.embed_image(image_path)
        return embedding

    def is_numeric_value(self,value):
        # Check if the value is an integer or a floating-point number
        if isinstance(value, (int, float)):
            return True
        else:
            return False

    def is_text_value(self,value):
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
        embedded_input=[]
        for feature in input:
            if self.is_image_value(feature):
                feature_embed = self.embed_image(feature)
                feature_embed = F.relu(self.img_reducer(feature_embed))

            elif self.is_text_value(feature):
                feature_embed = self.embed_text(feature)
                feature_embed = F.relu(self.text_reducer(feature_embed))
            elif self.is_numeric_value(feature):
                feature_embed = torch.tensor([feature], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            embedded_input.append(feature_embed)
        x = torch.cat(embedded_input, dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.logits(x)
        if logits.dim() == 2:
            probs = F.softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs


        return output

