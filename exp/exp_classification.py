import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from models import CNN, DLinear, TimesNet, Transformer, Reformer, Informer, Nonstationary_Transformer, KNN


class Exp_Classification():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'KNN': KNN,
            'CNN': CNN,
            'DLinear': DLinear,
            'Informer': Informer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'Reformer': Reformer,
            'TimesNet': TimesNet,
            'Transformer': Transformer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device) if self.args.model != 'KNN' else self._build_model()

    def _acquire_device(self):
        if self.args.use_gpu and self.args.model != 'KNN':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu and self.args.model != 'KNN':
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.args.model == 'KNN':
            return None
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model == 'KNN':
            return None
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        if self.args.model == 'KNN':
            val_features, val_labels = vali_data.to_numpy()
            accuracy = self.model.evaluate(val_features.squeeze(), val_labels.squeeze())
            return accuracy

        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        vali_data, vali_loader = self._get_data(flag='VAL')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.model == 'KNN':
            train_features, train_labels = train_data.to_numpy()
            self.model.fit(train_features.squeeze(), train_labels.squeeze())
            test_accuracy = self.vali(test_data, test_loader, criterion)
            print(f'Test Accuracy: {test_accuracy}')

            self.model.save_model(path)

            return self.model

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='VAL')
        if test:
            print('loading model')
            self.model.load_model('./checkpoints/' + setting)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        trues = trues.flatten().cpu().numpy()
        if self.args.model == 'KNN':
            predictions = preds.flatten().cpu().numpy()
        else:
            probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        
        accuracy = cal_accuracy(predictions, trues)

        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        def print_confusion_matrix(y_pred, y_test, labels, mode):
            conf_mat = confusion_matrix(y_pred, y_test)

            fig = plt.figure()
            fig.set_size_inches(7.0, 7.0)

            res = plt.imshow(np.array(conf_mat), cmap="OrRd", interpolation='nearest')
            for i, row in enumerate(conf_mat):
                for j, c in enumerate(row):
                    if c > 0:
                        plt.text(j - .2, i + .1, c, fontsize=16)

            fig.colorbar(res)
            plt.title('Confusion Matrix ' + mode)
            _ = plt.xticks(range(4), [l for l in labels.values()], rotation=90)
            _ = plt.yticks(range(4), [l for l in labels.values()])
            # plt.show()
            plt.savefig("plots/" + self.args.model + ".png")

        print_confusion_matrix(predictions, trues, {0: '0-67', 1: '67-230', 2: '230-4600', 3: '>4600'}, self.args.model)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
    
    def convert(self, setting):
        self.model.load_model('./checkpoints/' + setting)
        self.model.eval()

        self.model.to(self.device)

        input = torch.randn(1, self.args.seq_len, self.args.enc_in).to(self.device)

        # Define the path to save the ONNX model
        onnx_model_path = f'./checkpoints/{setting}/model.onnx'

        torch.onnx.export(
            self.model,
            input,
            onnx_model_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Model has been converted to ONNX format and saved at {onnx_model_path}")