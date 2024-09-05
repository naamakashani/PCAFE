from state_text import *
import torch.optim as optim
from RL.utils import *
import gymnasium
from transformers import AutoModel, AutoTokenizer
from guesser_textual import *

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


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





class myEnv(gymnasium.Env):

    def __init__(self,
                 flags,
                 device):
        self.device = device
        self.X, self.y, self.features_size= load_existing_image_data()
        self.num_classes= len(np.unique(self.y))


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.3)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.05)
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = ImageEmbedder()
        self.text_embedding_dim = self.text_model.config.hidden_size
        self.embedding_dim = 64



        self.guesser = Guesser(self.embedding_dim + self.text_embedding_dim, self.num_classes)
        self.question_embedding = nn.Embedding(num_embeddings=self.features_size, embedding_dim=self.embedding_dim)

        self.state = State(self.features_size, self.embedding_dim + self.text_embedding_dim, self.device)
        cost_list = np.array(np.ones(self.features_size + 1))
        self.action_probs = torch.from_numpy(np.array(cost_list))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_guesser = optim.Adam(self.guesser.parameters(), lr=flags.lr)
        self.optimizer_state = optim.Adam(self.state.parameters(), lr=flags.lr)
        self.optimizer_embedding = optim.Adam(self.question_embedding.parameters(), lr=flags.lr)
        self.episode_length = 1
        self.count = 0

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True):

        # Reset state
        self.state.reset_states()
        self.s = self.state.lstm_h.data.cpu().numpy()

        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.s = np.array(self.s)
        self.time = 0
        if mode == 'training':
            self.train_guesser = train_guesser
        else:
            self.train_guesser = False
        return self.s

    def reset_mask(self):
        """ A method that resets the mask that is applied
        to the q values, so that questions that were already
        asked will not be asked again.
        """
        mask = torch.ones(self.features_size + 1)
        mask = mask.to(device=self.device)
        return mask

    def step(self,
             action, mask
             , i, mode='training'):
        """ State update mechanism """

        # update state
        next_state = self.update_state(action, mode, mask, i)
        self.s = next_state

        # compute reward
        self.reward = self.compute_reward(mode)

        self.time += 1
        if self.time == self.features_size:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess

    # Update 'done' flag when episode terminates
    def terminate_episode(self):
        self.done = True

    def prob_guesser(self, state):
        guesser_input = torch.Tensor(
            state[:self.features_size])
        if torch.cuda.is_available():
            guesser_input = guesser_input.cuda()
        self.guesser.train(mode=False)
        self.probs = self.guesser(guesser_input)
        self.probs = F.softmax(self.probs, dim=1)
        self.guess = torch.argmax(self.probs).item()
        class_index = int(self.y_train[self.patient].item())
        self.correct_prob = self.probs[0, class_index].item()
        return self.correct_prob

    def print_parametrs(self):
        # Print parameters of guesser after optimization step
        print("\nGuesser parameters after optimization step:")
        for param_name, param in self.guesser.named_parameters():
            print(param_name, param)

        # Print parameters of state after optimization step
        print("\nState parameters after optimization step:")
        for param_name, param in self.state.named_parameters():
            print(param_name, param)

        # Print parameters of question_embedding after optimization step
        print("\nQuestion Embedding parameters after optimization step:")
        for param_name, param in self.question_embedding.named_parameters():
            print(param_name, param)

    def embed_text(self, text):
        tokens = self.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = self.text_model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming you want to use the [CLS] token
        return embeddings


    def embed_image(self, image_path):
        embedding = self.embedder.embed_image(image_path)
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
    def is_image_value(self,value):
       #check if value is path that ends with 'png' or 'jpg'
        if isinstance(value, str):
            if value.endswith('png') or value.endswith('jpg'):
                return True
            else:
                return False

    def update_state(self, action, mode, mask, eps):
        prev_state = self.s
        if action < self.features_size:  # Not making a guess
            if mode == 'training':
                answer = self.X_train[self.patient, action]
            elif mode == 'val':
                answer = self.X_val[self.patient, action]
            elif mode == 'test':
                answer = self.X_test[self.patient, action]
            #check type of feature
            if self.is_numeric_value(answer):
                answer_vec = torch.unsqueeze(torch.ones(self.text_embedding_dim) * answer, 0)
            elif self.is_image_value(answer):
                answer_vec = self.embed_image(answer)
            elif self.is_text_value(answer):
                answer_vec = self.embed_text([answer])

            ind = torch.LongTensor([action]).to(device=self.device)
            question_embedding = self.question_embedding(ind)
            question_embedding = question_embedding.to(device=self.device)
            next_state = self.state(question_embedding, answer_vec)
            next_state = torch.autograd.Variable(torch.Tensor(next_state))
            next_state = next_state.float()
            probs = self.guesser(next_state)
            y_true = self.y_train[self.patient]
            y_tensor = torch.tensor([int(y_true)])
            y_true_tensor = F.one_hot(y_tensor, num_classes=self.num_classes)
            self.probs = probs.float()
            self.probs = F.softmax(self.probs, dim=1)
            y_true_tensor = y_true_tensor.float()
            self.optimizer_state.zero_grad()
            self.optimizer_guesser.zero_grad()
            self.optimizer_embedding.zero_grad()
            self.loss = self.criterion(self.probs, y_true_tensor)
            self.loss.backward()
            if eps >= 0:
                self.count = self.count + 1
                self.optimizer_state.step()
                self.optimizer_guesser.step()
                self.optimizer_embedding.step()
            self.reward = self.prob_guesser(next_state) - self.prob_guesser(prev_state)
            self.guess = -1
            self.done = False
            return next_state

        else:
            self.reward = self.prob_guesser(prev_state)
            self.terminate_episode()
            return prev_state



    # def update_state(self, action, mode, mask, eps):
    #     prev_state = self.s
    #
    #     if action < self.features_size:  # Not making a guess
    #         if self.
    #         if action in self.text_index:
    #             index = self.text_index.index(action)
    #             if mode == 'training':
    #                 answer = self.text_train[self.patient, index]
    #             if mode == 'val':
    #                 answer = self.text_val[self.patient, index]
    #             if mode == 'test':
    #                 answer = self.text_test[self.patient, index]
    #             answer_vec = self.embed_text([answer])
    #         if action in self.image_index:
    #             index = self.image_index.index(action)
    #             if mode == 'training':
    #                 answer = self.image_features[self.patient, index]
    #             if mode == 'val':
    #                 answer = self.image_features[self.patient, index]
    #             if mode == 'test':
    #                 answer = self.image_features[self.patient, index]
    #             answer_vec= self.embed_image(answer)
    #         else:
    #             index = self.numeric_index.index(action)
    #             if mode == 'training':
    #                 answer = self.X_train[self.patient, index]
    #             elif mode == 'val':
    #                 answer = self.X_val[self.patient, index]
    #             elif mode == 'test':
    #                 answer = self.X_test[self.patient, index]
    #             answer_vec = torch.unsqueeze(torch.ones(self.text_embedding_dim) * answer, 0)
    #
    #         ind = torch.LongTensor([action]).to(device=self.device)
    #         question_embedding = self.question_embedding(ind)
    #         question_embedding = question_embedding.to(device=self.device)
    #         next_state = self.state(question_embedding, answer_vec)
    #         next_state = torch.autograd.Variable(torch.Tensor(next_state))
    #         next_state = next_state.float()
    #         probs = self.guesser(next_state)
    #         y_true = self.y_train[self.patient]
    #         y_tensor = torch.tensor([int(y_true)])
    #         y_true_tensor = F.one_hot(y_tensor, num_classes=2)
    #         self.probs = probs.float()
    #         self.probs = F.softmax(self.probs, dim=1)
    #         y_true_tensor = y_true_tensor.float()
    #         self.optimizer_state.zero_grad()
    #         self.optimizer_guesser.zero_grad()
    #         self.optimizer_embedding.zero_grad()
    #         self.loss = self.criterion(self.probs, y_true_tensor)
    #         self.loss.backward()
    #         if eps >= 0:
    #             self.count = self.count + 1
    #             self.optimizer_state.step()
    #             self.optimizer_guesser.step()
    #             self.optimizer_embedding.step()
    #         self.reward = self.prob_guesser(next_state) - self.prob_guesser(prev_state)
    #         self.guess = -1
    #         self.done = False
    #         return next_state
    #
    #     else:
    #         self.reward = self.prob_guesser(prev_state)
    #         self.terminate_episode()
    #         return prev_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None

        if self.guess == -1:  # no guess was made
            return self.reward

        if mode == 'training':
            y_true = self.y_train[self.patient]
            if self.train_guesser:
                self.guesser.optimizer.zero_grad()
                self.guesser.train(mode=True)
                y_tensor = torch.tensor([int(y_true)])
                y_true_tensor = F.one_hot(y_tensor, num_classes=self.num_classes).squeeze()
                self.probs = self.probs.float()
                y_true_tensor = y_true_tensor.float()
                self.guesser.loss = self.guesser.criterion(self.probs, y_true_tensor)
                self.guesser.loss.backward()
                self.guesser.optimizer.step()

        return self.reward
