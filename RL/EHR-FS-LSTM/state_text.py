from agent_text import *
from guesser_textual import *

class State(nn.Module):

    def __init__(self, features_size, concat_dim, device):
        super(State, self).__init__()
        self.device = device
        self.features_size = features_size
        self.concat_dim = concat_dim
        self.lstm = nn.LSTMCell(input_size=self.concat_dim , hidden_size=self.concat_dim)
        self.initial_c = nn.Parameter(torch.randn(1, self.concat_dim ), requires_grad=True).to(
            device=self.device)
        self.initial_h = nn.Parameter(torch.randn(1, self.concat_dim ), requires_grad=True).to(
            device=self.device)
        self.reset_states()

    def reset_states(self):
        self.lstm_h = (torch.zeros(1, self.concat_dim) + self.initial_h).to(device=self.device)
        self.lstm_c = (torch.zeros(1, self.concat_dim ) + self.initial_c).to(device=self.device)

    def forward(self, question_encode, answer_vec):
        question_embedding = question_encode.to(device=self.device)
        answer_vec = answer_vec.to(device=self.device)
        x = torch.cat((question_embedding,answer_vec), dim=1)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h, self.lstm_c))
        return self.lstm_h.data.cpu().numpy()
