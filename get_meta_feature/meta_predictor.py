import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from utils import EarlyStopping, adjust_learning_rate, get_device, set_seed
import os
import numpy as np

class meta_predictor_MLP(nn.Module):
    def __init__(self, meta_feature_dim, n_models, embedding_dim=5,dropout_rate=0.1):
        super(meta_predictor_MLP, self).__init__()

        self.embeddings = nn.Embedding(n_models, embedding_dim)

        self.predictor = nn.Sequential(
            nn.Linear(meta_feature_dim+embedding_dim+1, 2*(meta_feature_dim+embedding_dim+1)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2*(meta_feature_dim+embedding_dim+1), 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid())

    def forward(self, meta_features=None, pred_len=None, model_id=None):
        # print(model_id)
        model_embedding = self.embeddings(model_id)
        pred_len = pred_len.reshape(-1,1)
        embedding = torch.cat((meta_features, pred_len, model_embedding), dim=1)
        pred = self.predictor(embedding)

        return pred

class meta_predictor_dl(nn.Module):
    def __init__(self,model_name, meta_feature_dim, n_models, learning_rate, n_epochs, optimizer, lradj, train_iter, loss_fn,patience=7, pct_start=0.2, embedding_dim=5,dropout_rate=0.1):
        super(meta_predictor_dl,self).__init__()
        if model_name =='mlp':
            self.model = meta_predictor_MLP(meta_feature_dim, n_models, embedding_dim,dropout_rate)
        else:
            raise NotImplementedError
        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError
        self.lradj = lradj
        if lradj == 'COS':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8)
        else:
            self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                steps_per_epoch=train_iter,
                                                pct_start=pct_start,
                                                epochs=n_epochs,
                                                max_lr=learning_rate)

        self.n_epochs = n_epochs

        self.early_stopping = EarlyStopping(patience=patience)
        self.device = get_device(use_gpu=True)
        
    def fit(self,train_loader,checkpoint_path):
        print(os.path.join(checkpoint_path,'checkpoint'))
        if os.path.exists(os.path.join(checkpoint_path,'checkpoint')):
            print('model checkpoint file exist!')
            checkpoint_file = torch.load(os.path.join(checkpoint_path,'checkpoint'))
            self.model.load_state_dict(checkpoint_file)
            self.model = self.model.to(self.device)
            return self
        self.model = self.model.to(self.device)
        for epoch in range(self.n_epochs):
            train_loss = []
            self.model.train()
            for i,(meta_feature,pred_len,model_id,y_true) in enumerate(train_loader):
                self.optimizer.zero_grad()

                meta_feature = meta_feature.float().to(self.device)
                pred_len = pred_len.float().to(self.device)
                model_id = model_id.int().to(self.device)
                y_true = y_true.float().to(self.device)
                y_pred = self.model(meta_feature,pred_len,model_id)
                loss = self.loss_fn(y_pred,y_true)
                train_loss.append(loss)
                loss.backward()
                self.optimizer.step()

                if self.lradj == 'TST':
                    adjust_learning_rate(self.optimizer,scheduler,epoch+1,self.lradj,self.learning_rate,printout=True)
                    self.scheduler.step()
            
            train_loss = sum(train_loss)/len(train_loss)
            print(f'Epoch: {epoch+1} | Train Loss:{train_loss}.')

            self.early_stopping(train_loss, self.model, checkpoint_path)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.lradj != 'TST':
                if self.lradj == 'COS':
                    self.scheduler.step()
                    print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        self.learning_rate = self.optimizer.param_groups[0]['lr']
                        print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
                    adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.lradj,self.learning_rate, printout=True)
            else:
                print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()[0]))

        return self
    
    def predict(self, test_meta_feature, test_pred_len, test_method):
        self.model.eval()
        self.model = self.model.cpu()
        test_meta_feature = test_meta_feature.float().cpu()
        test_pred_len = test_pred_len.float().cpu()
        test_method = test_method.int().cpu()
        
        with torch.no_grad():
            y_pred = self.model(test_meta_feature, test_pred_len, test_method)
        
        return y_pred