import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file :The id in does not start from 0, so it needs to be remapped, id_filename is the order of the nodes
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # Map the node id（idx）to an index starting from 0

            with open(distance_df_filename, 'r') as f:
                f.readline()  # Skip the header line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance file The id in starts directly from 0

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance fileThe id in does not start from 0, so it needs to be remapped；id_filename Is the order of the nodes
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # Map the node id (idx) to an index starting from 0

            with open(distance_df_filename, 'r') as f:
                f.readline()  # Skip the header line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file The id in starts directly from 0

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA

def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # First make sure that A is a symmetric matrix

    D = np.diag(np.sum(A, axis=1))  # D is a degree matrix, only the diagonal has elements

    L = D - A  # L is a real symmetric matrix A, and the eigenvectors corresponding to n different eigenvalues are orthogonal.

    return L

def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # Add self-connection to neighbor matrix
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # Add self-connection to the adjacency matrix
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def compute_val_loss(net, val_loader, criterion, sw, epoch):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # Record the loss of all batches

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B，N，T，1)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # Only take the first value entered as input，After that, the value from predict is used as input
            decoder_input_list = [decoder_start_inputs]
            #Forecast by time step
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)  # Calculation error (MSELoss)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # Store all batch inputs

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # Only take the first value entered as input，After that, the value from predict is used as input
            decoder_input_list = [decoder_start_inputs]

           # Forecast according to time steps
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # Calculation error
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    Process both x and y into data before normalization to [-1,1];
    Each sample contains the data of all monitoring points at the same time, so the data constructed by this function is input into the spatio-temporal sequence prediction model;
    This function will string together hour, day, week time;
    Note: For the data read from the file, x and y are all normalized values
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

   # Uniformly normalize y to a value between [-1,1]
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),Finally, the traffic flow is known as the initial input of the decoder
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),Finally, the traffic flow is known as the initial input of the decoder
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),Finally, the traffic flow is known as the initial input of the decoder
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min


