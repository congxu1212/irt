import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, llm_input_dim, item_input_dim, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.item_dim = item_input_dim
        self.llm_dim = llm_input_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 512  # changeable

        super(Net, self).__init__()
        # prediction sub-net
        self.model_emb = nn.Linear(self.llm_dim, self.knowledge_dim)
        self.k_difficulty = nn.Linear(self.item_dim, self.knowledge_dim)
        self.k = nn.Linear(self.knowledge_dim, self.knowledge_dim)
        self.e_difficulty = nn.Linear(self.item_dim, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len2)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        self.softmax = nn.Softmax(dim=1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, llm, input_query, input_knowledge_point):
        # before prednet
        llm_emb = self.model_emb(llm)
        stat_emb = torch.sigmoid(llm_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_query))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_query)) * 9
        # ############################
        # ############################
        if len(input_knowledge_point.shape) == 1:
            input_knowledge_point = input_knowledge_point.unsqueeze(0)  
        input_knowledge_point = self.softmax(input_knowledge_point)
        # ############################
        # ############################
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.squeeze(), stat_emb, e_difficulty, k_difficulty, input_knowledge_point


class NIRT():
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, llm_dim, item_dim, knowledge_n):
        super(NIRT, self).__init__()
        self.nirt_net = Net(llm_dim, item_dim, knowledge_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.nirt_net = self.nirt_net.to(device)
        self.nirt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.nirt_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                llm, item, knowledge_emb, y = batch_data
                llm: torch.Tensor = llm.to(device)
                item: torch.Tensor = item.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                rmse, mae, auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] rmse: %.6f, mae: %.6f, auc: %.6f, accuracy: %.6f" % (epoch_i, rmse, mae, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        self.nirt_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            llm, item, knowledge_emb, y = batch_data
            llm: torch.Tensor = llm.to(device)
            item: torch.Tensor = item.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
            y: torch.Tensor = y.to(device)
            loss = loss_function(pred, y)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), roc_auc_score(np.array(y_true)>= 0.5, np.array(y_pred)>= 0.5), accuracy_score(np.array(y_true) >= 0.5, np.array(y_pred) >= 0.5)

    def generate(self, llm, item, knowledge_emb, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
        return pred.tolist()
    
    def get_theta(self, llm, item, knowledge_emb, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
        return theta.tolist()
    
    def get_e(self, llm, item, knowledge_emb, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
        return a.tolist()
    
    def get_difficulty(self, llm, item, knowledge_emb, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
        return b.tolist()
    
    def get_relevance(self, llm, item, knowledge_emb, device="cpu"):
        self.nirt_net = self.nirt_net.to(device)
        llm: torch.Tensor = llm.to(device)
        item: torch.Tensor = item.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        pred, theta, a, b, r = self.nirt_net(llm, item, knowledge_emb)
        return r.tolist()

    def save(self, filepath):
        torch.save(self.nirt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.nirt_net.load_state_dict(torch.load(filepath, map_location="cuda"))
        logging.info("load parameters from %s" % filepath)
