from typing import Text
from torch import nn
from torch import functional as F

class TextClassificationCNN(nn.Module):

    def __init__(self, vocab_size=10000, embed_dim=100, num_filter=100, filter_length=3, num_class=2):
        super(TextClassificationCNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.conv1 = nn.Conv1d(embed_dim, num_filter, filter_length)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x = F.relu(self.conv1(x))
        x = self.fc(x)
        return x



if __name__ == '__main__':

    CNN_model = TextClassificationCNN(vocab_size=10000, embed_dim=100, num_class=2)
    print(CNN_model)