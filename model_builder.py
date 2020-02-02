import torch
import torch.nn as nn
from tqdm import tqdm
from preprocessing import Data_Sat
import matplotlib.pyplot as plt
import numpy as np
from torch import FloatTensor

class LSTM(nn.Module):
    def __init__(self, device, input_dim=7, output_dim=6, lstm_hidden_dim=20,
                 lstm_layers_count=1, bidirectional=False, dropout=0,
                 previous_hid_state=True):
        super().__init__()
        self.device = device
        self.previous_hid_state = previous_hid_state
        self.input_dim = input_dim
        self.lstm_layers_count = lstm_layers_count
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size = self.input_dim,
                            hidden_size = self.lstm_hidden_dim,
                            num_layers = self.lstm_layers_count,
                            bidirectional=bidirectional,
                            bias=True,
                            dropout=dropout
                           )

        self.linear = nn.Linear(lstm_hidden_dim*(1+bidirectional), output_dim, bias=True)

    def init_hidden(self, batch_size):
        self.h = torch.zeros(self.lstm_layers_count * (2 if self.bidirectional else 1),
                             batch_size, self.lstm_hidden_dim).to(self.device)
        self.c = torch.zeros(self.lstm_layers_count * (2 if self.bidirectional else 1),
                             batch_size, self.lstm_hidden_dim).to(self.device)



    def forward(self, inputs):
        if not self.previous_hid_state:
            self.init_hidden(inputs.shape[1])
        lstm_out, (self.h, self.c) = self.lstm.forward(inputs, (self.h, self.c))
        linear_out = self.linear.forward(lstm_out)
        self.h = self.h.detach()
        self.c = self.c.detach()

        return linear_out

def smape(satellite_predicted_values, satellite_true_values):
    # the division, addition and subtraction are point twice
    return torch.mean(torch.abs(satellite_predicted_values - satellite_true_values)
        / (torch.abs(satellite_predicted_values) + torch.abs(satellite_true_values)))

def do_epoch(model, loss_function, data, batch_size, optimizer=None, name=None):
    """
       Генерация одной эпохи
    """
    epoch_loss = 0

    max_sequence_count, sequence_length = data[0].shape[0], data[0].shape[1]
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    batch_count = len(loader)

    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batch_count) as progress_bar:
            for i, sample in enumerate(loader):
                # перестановка осей. Стало - [max_sequence_count, sequence_length, batch_size, values]
                sample = sample.permute(1, 2, 0, 3)
                model.init_hidden(sample.shape[2])
                for sequence in sample:
                    X_batch, y_batch = (sequence[..., :7]).to(model.device.type), (sequence[..., 7:]).to(model.device.type)
                    prediction = model(X_batch)

                    loss = smape(prediction, y_batch)  # используем целевую метрику в качестве Loss
                    epoch_loss += loss.item()

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(
                    name, loss.item())
                )

            epoch_loss /= (i + 1) * max_sequence_count
            score = float((1-epoch_loss) * 100)

            progress_bar.set_description(f'Epoch {name} -  score: {score:.2f}')

    return score

def fit(model, loss_function, batch_size=1, epochs_count=10, optimizer=None,
        scheduler=None, train_dataset=None, val_dataset=None, plot_draw=False):
    """
    Функция тренировки
    return: (list) значение Score для Train и Val на каждой эпохе
    """
    train_history = []
    val_history = []
    for epoch in range(epochs_count):
            name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
            epoch_train_score = 0
            epoch_val_score = 0
            if train_dataset:
                epoch_train_score = do_epoch(model, loss_function, train_dataset, batch_size,
                                              optimizer, name_prefix + 'Train:')
                train_history.append(epoch_train_score)

            if val_dataset:
                name = '  Val:'
                if not train_dataset:
                    name = ' Test:'
                epoch_val_score = do_epoch(model, loss_function, val_dataset, batch_size,
                                             optimizer=None, name=name_prefix + name)
                val_history.append(epoch_val_score)

                scheduler.step(epoch_val_score)
            else:
                scheduler.step(epoch_train_score)



    if plot_draw:
            draw_plot(train_history, val_history)

    return train_history, val_history

def cross_validation(model, data, folds, loss_function, sequence_length, max_sequence_count, optimizer=None, scheduler=None,
        epochs_count=1, batch_size=1, plot_draw=False):
    """
    тренировка модели с кросс-валидацией и валидацией после каждой эпохи, валидация есть по умолчанию.
    Выводит списки fold_train_history fold_val_history.
    """

    fold_train_history = []
    fold_val_history = []

    optim_default_state = optimizer.state_dict() if optimizer else None
    sched_default_state = scheduler.state_dict() if scheduler else None

    for j, fold in enumerate(folds):

        #Возврат оптимизатора к изначальным значениям
        if optimizer:
            optimizer.load_state_dict(optim_default_state)

        #Scheduler на каждом фолде заново определяется
        if scheduler:
            scheduler.load_state_dict(sched_default_state)

        #Сброс параметров модели на каждом фолде
        for name, module in model.named_children():
            print('resetting ', name)
            module.reset_parameters()

        print('Fold: ', j+1, '\n')

        val_data = data.loc[fold]
        val_dataset = Data_Sat(model.device, val_data, sequence_length)
        val_dataset.generate_samples(max_sequence_count=max_sequence_count,  last_sequence=False)

        train_data = data.loc[[index for nfold in folds for index in nfold if nfold != fold]]
        train_dataset = Data_Sat(model.device, train_data, sequence_length)
        train_dataset.generate_samples(max_sequence_count=max_sequence_count, last_sequence=False)



        train_history, val_history = fit(model, loss_function, batch_size, epochs_count, optimizer, scheduler,
                                         train_dataset, val_dataset, plot_draw)

        fold_val_history.append(val_history[-1])
        fold_train_history.append(train_history[-1])
    return fold_train_history, fold_val_history

def draw_plot(train_loss_history, val_loss_history):
    """
    Рисует lineplot
    """
    data = pd.DataFrame(data=[train_loss_history, val_loss_history], index=['Train', 'Val']).T
    plt.figure(figsize=(15, 6))
    sns.set(style='darkgrid')
    ax = sns.lineplot(data=data, markers = ["o", "o"], palette='bright')
    plt.title("Line Plot", fontsize = 20)
    plt.xlabel("Epoch", fontsize = 15)
    plt.ylabel("Loss", fontsize = 15)
    plt.show()

def predict(model, sat_data):
    """
    Получает на вход модель и разделенные на sequences_count, sequence_length данные. Предсказывает реальные значение по спутнику.
    Выводит Tensor формы (n_samples, n_features).
    """
    sequences_count, sequence_length, _ = sat_data.shape
    result = torch.zeros((sequences_count*sequence_length, 6)).to(model.device)
    model.eval()
    model.init_hidden(1)
    for i, seq in enumerate(sat_data):
        inputs = FloatTensor(seq[:, None, :]).to(model.device)
        predicted = model(inputs)

        predicted = predicted.view(sequence_length, -1).detach()
        result[i*sequence_length : (i+1)*sequence_length] = predicted
    return result

def save_model(path, model, optimizer, scheduler, train_history, val_history):
    torch.save({
            'epoch': len(train_history),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_history': train_history,
            'val_history': val_history
            }, path)
    print('successfully saved')

def load_model(path, model, optimizer, scheduler, train_history, val_history):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_history = checkpoint['train_history']
    val_history = checkpoint['val_history']
    print('successfully loaded')
