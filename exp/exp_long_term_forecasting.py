from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import shutil
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if 'Gym' not in self.args.model:
            model = self.model_dict[self.args.model].Model(self.args).float()
            self.save_suffix = ''
        else:
            model_name, gym_x_mark, gym_series_sampling, gym_series_norm, gym_series_decomp, \
            gym_channel_independent, gym_input_embed, gym_network_architecture, gym_attn, gym_feature_attn, \
            gym_encoder_only, gym_frozen = self.args.model.split('_')
            model = self.model_dict[model_name].Model(self.args,
                                                      gym_x_mark=gym_x_mark,
                                                      gym_series_sampling=gym_series_sampling,
                                                      gym_series_norm=gym_series_norm,
                                                      gym_series_decomp=gym_series_decomp,
                                                      gym_channel_independent=gym_channel_independent,
                                                      gym_input_embed=gym_input_embed,
                                                      gym_network_architecture=gym_network_architecture,
                                                      gym_attn=gym_attn,
                                                      gym_feature_attn=gym_feature_attn,
                                                      gym_encoder_only=gym_encoder_only,
                                                      gym_frozen=gym_frozen).float()
            self.save_suffix = 'Gym'

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=list(range(len(self.args.device_ids))))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.args.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.args.loss == 'HUBER':
            criterion = nn.HuberLoss(delta=0.5)
        else:
            raise NotImplementedError
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(f'{self.args.checkpoints}{self.save_suffix}/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path) and False: # TODO: 暂时不跳过实验
            print(f'The model file already exists! loading...')
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            epoch_time_avg = []
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                    iter_count += 1
                    model_optim.zero_grad()

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                epoch_time_avg.append(time.time() - epoch_time)
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)
  
            self.train_cost = np.mean(epoch_time_avg)
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        checkpoint_path = f'./checkpoints{self.save_suffix}/' + setting
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))

        preds, trues = [], []
        # folder_path = f'./test_results{self.save_suffix}/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        dataset = self.args.model_id.split('_')[0]
        if 'TSGym' in setting:
            if 'Transformer' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_transformer/{dataset}/' + setting + '/'
            elif 'LLM' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_LLM/{dataset}/' + setting + '/'
            elif 'TSFM' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_TSFM/{dataset}/' + setting + '/'
            else:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_non_transformer/{dataset}/' + setting + '/'
        else:
            folder_path = f'./results_long_term_forecasting/results{self.save_suffix}/{dataset}/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, self.train_cost]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)

        return