from torch.utils.data import Dataset
import numpy as np
import math
from torch import FloatTensor

class Norm():
    """
    Нормализатор.
    Init запоминает среднее и стандартное отклонение в данных
    """
    def __init__(self, df, ignore_column=None):
        self.mean = df.mean()
        self.std = df.std()
        self.l2 = None
        self._get_l2(df)
        if ignore_column:
            self.mean[ignore_column] = 0
            self.std[ignore_column] = 1
            self.l2[ignore_column] = 1

    def _get_l2(self, df):
        self.l2 = df.pow(2, axis=0).sum(axis=0).pow(0.5, axis=0) / df.shape[0]**0.5
        l2_dict = {
                'Vx' : 2.63748924855871,
                'Vy' : 2.6003214462464426,
                'Vz' : 2.113766985332456,
                'x' : 25391.823604180147,
                'y' : 25609.50935919694,
                'z' : 20668.126741013515,
                'delta_seconds' : 3586.29103822237}

        for c in self.l2.index:
            saved_l2 = c.replace('_sim', '')
            if saved_l2 in l2_dict.keys():
                self.l2[c] = l2_dict[saved_l2]

    @staticmethod
    def columns_check(columns, df_columns):
        if not columns:
            return df_columns
        return columns

    def z_norm(self, df, columns=None):
        columns = self.columns_check(columns, df.columns)
        return (df[columns] - self.mean[columns]) / self.std[columns]

    def l2_norm(self, df, columns=None):
        columns = self.columns_check(columns, df.columns)

        return df[columns] / self.l2[columns]

    def back_z_norm(self, df, columns=None):
        try:
            columns = self.columns_check(columns, df.columns)
        except:
            print("df должен быть DataFrame или columns должен быть заполнен")
            return None
        if not type(df) is pd.core.frame.DataFrame:
            df = pd.DataFrame(data=df, columns=columns)

    def back_l2_norm(self, df, columns=None):
        try:
            columns = self.columns_check(columns, df.columns)
        except:
            print("df должен быть DataFrame или columns должен быть заполнен")
            return None
        if not type(df) is pd.core.frame.DataFrame:
            df = pd.DataFrame(data=df, columns=columns)

        return df[columns] * self.l2[columns]

class Data_Sat(Dataset):
    def __init__(self, device, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.data = data
        self.satellite_dict = {}
        self.split_data()
        self.device = device

    def split_data(self):
        # разделяет данные по каждому спутнику на отдельные секвенции длиной sequence_length каждая
        # и записывает их в словарь self.satellite_dict
        # Если значений не хватило до sequence то дописывает нули

        for ind, satellite in enumerate(self.data['sat_id'].unique()):
            # берем данные по одному спутнику начиная со столбца delta_seconds (нулевой столбец sat_id пропускаем)
            sat_data = self.data.query('sat_id==@satellite').loc[:, 'delta_seconds':]
            sequence_count = int(math.ceil(sat_data.shape[0] / self.sequence_length))
            samples_sat = np.zeros((sequence_count * self.sequence_length, sat_data.shape[1]))
            samples_sat[: sat_data.shape[0]] = sat_data.values
            self.satellite_dict[ind] = samples_sat.reshape(sequence_count, self.sequence_length, -1)

    def generate_samples(self, max_sequence_count=100, last_sequence=False):
        # генерирует отдельные наборы последовательных секвенций, аугментируя данные:
        # разбивает данные по одному спутнику (если sequence_count больше чем max_sequence_count)
        # на несколько отдельных последовательностей длиной max_sequence_count
        # например спутник с размером (sequence_count=200, sequence=20,...)
        # функция преобразует в 2 спутника размером (max_sequence_count=100, sequence=20, ...)
        # last_sequence=False - не добавляет последнюю sequence
        self.samples = []

        for sat in self.satellite_dict.values():
            sequence_count = sat.shape[0]
            if not last_sequence:
                sequence_count -= 1
            if  sequence_count > max_sequence_count:
                samples_count = math.ceil(sequence_count / max_sequence_count)
                step = (sequence_count - max_sequence_count) / (samples_count - 1)
                for sample in range(samples_count):
                    next_step = round(step * sample)
                    self.samples.append(self.data_casting(sat[next_step: next_step + max_sequence_count]))

    @staticmethod
    def data_casting(data):
        # вычитает из значений симуляции начальную ошибку.
        # начальная ошибка равна x_sym[0] - x[0] и аналогично для y, z и т.д.
        for i in range(1, 7, 1):
            data[..., i + 6] -= data[0, 0, i + 6] - data[0, 0, i]
        return data


    def predict_to_df(self, predicts):
        # Переводит predict в датафрейм и добавляет id
        self.result = pd.DataFrame()

        for sequense in predicts:
            filters = np.abs(sequense).sum(axis=1) != 0
            self.result = self.result.append(pd.DataFrame(sequense[filters]), ignore_index=True)

        assert self.result.shape[0] == self.data.shape[0]
        self.result['id'] = (self.data['id'].values).astype('int64')
        self.result = self.result[['id', 0, 1, 2, 3, 4, 5]]
        self.result.columns = ['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']

    def __len__(self):
        """
        Returns total number of samples
        """
        return len(self.samples)

    def __getitem__(self, index):
        """

        :param index:
        :return: one-satellite sample [max_sequence_count, sequence_length, gt + in values]
        """
        return FloatTensor(self.samples[index]).to(self.device)

def split_data(values, coeff=0.9):
    # coeff - доля спутников для тренировки
    split = int(np.floor(coeff * values))
    indices = list(range(values))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    return train_indices, test_indices

def split_folds(indices, n_folds):
    # делит список индексов на n_folds частей
    avg = len(indices) / float(n_folds)
    result = []
    last = 0
    for _ in range(n_folds):
        result.append(indices[int(last):int(last + avg)])
        last += avg
    return result

def add_delta_time(df, columns=None):
    """
    Добавляет столбец delta_time в секундах. Возвращает DataFrame в порядке указанном columns
    если columns нет то возвращает все столбцы
    """
    dataframe = df.sort_values(by=['sat_id', 'epoch'], inplace=False)
    dataframe['delta_time'] = dataframe.iloc[1:]['epoch'] - dataframe.iloc[0:-1]['epoch'].values
    dataframe['delta_seconds'] = dataframe['delta_time'].dt.seconds
    filters = dataframe['sat_id'] != np.insert(dataframe.iloc[0:-1]['sat_id'].values, 0, -1)
    dataframe.loc[filters, ['delta_time', 'delta_seconds']] = 0
    if not columns:
        columns=dataframe.columns
    return dataframe[columns]
