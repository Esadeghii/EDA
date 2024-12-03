from contextlib import nullcontext
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod
import torch
from torch import nn  # Q: is this necessary to do?

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.helpers import to_numpy  # Disregard syntax errors on this line
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder

GmmData = pd.read_excel('data-and-cleaning/GMMdata.xlsx')
GmmDData = pd.read_excel('data-and-cleaning/CSdistance.xlsx')
print("Distances Data loaded")
ALPHABET = ['A','C','G','T']
GmmDataS = GmmData['Sequence'].to_list()
clusternums = len(set(GmmData['Label'].to_list()))
enc = OneHotEncoder()
enc.fit(np.array(ALPHABET).reshape(-1,1))
# GmmDData = np.zeros((2099,2099))
# GMMSums = sum(sum(GmmDData))#.to_numpy()))
GMMSums = sum(sum(GmmDData.to_numpy()))
scale = 10


#read corresponding sequence and return the distance matrix
def Distencecs(sequence):
    basesequence = [''.join(enc.inverse_transform(e).reshape(-1).tolist()) for e in sequence] 
    filterdDdata = np.zeros((len(basesequence),len(basesequence)))
    for idx,clmns in enumerate(basesequence):
        for jdx,rows in enumerate(basesequence):
            if np.isnan(GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]):
                raise ValueError('GmmDData is nan')
            filterdDdata[idx][jdx] =  GmmDData[GmmDataS.index(clmns)][GmmDataS.index(rows)]


    return filterdDdata


def rbf(inputs,delta):
    return torch.exp(- inputs ** 2 / (2. * delta ** 2))


all_label_dis_sum_train = [0]*clusternums
all_label_dis_sum_val = [0]*clusternums
clusters_distance_latent_train = []
clusters_distance_latent_valid = []
all_label_code = []
  
class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(  # Adam is an alternate method for minimizing loss function
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.global_iter = 0
        self.trainer_config = ''
        self.writer = None
    

    def train_model(self, batch_size, num_epochs, log=False, weightedLoss=False):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """
        # set-up log parameters
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboardX summary writer
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + self.model.__repr__() + st)
            )

        # get dataloaders
        (generator_train,
         generator_val,
         _,) = self.dataset.data_loaders(
            batch_size=batch_size
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        print('Uniqe Label Train: ', set(self.dataset.train_label.reshape(-1)))
        print('Uniqe Label Valid: ', set(self.dataset.val_label.reshape(-1)))

        train_seq = np.array([itm[0].detach().numpy() for itm in generator_train.dataset])
        val_seq = np.array([itm[0].detach().numpy() for itm in generator_val.dataset])
    
        # create distance matrix for validation data & train data
        distance_train = Distencecs(train_seq)
        distance_val = Distencecs(val_seq)
        print("validation distance clusters Created")

        distence = []
        distence_mun = []

        correlation = []
        correlation_valid = []
        # train epochs
        for epoch_index in range(num_epochs):
            # update training scheduler
            self.update_scheduler(epoch_index)

            # run training loop on training data
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True,
                weightedLoss=weightedLoss
            )

            # run evaluation loop on validation data
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False,
                weightedLoss=weightedLoss
            )

            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index,
            )

            # log parameters
            if log:
                # log value in tensorboardX for visualization
                self.writer.add_scalar('loss/train', mean_loss_train, epoch_index)
                self.writer.add_scalar('loss/valid', mean_loss_val, epoch_index)
                self.writer.add_scalar('acc/train', mean_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/valid', mean_accuracy_val, epoch_index)

            # print epoch stats
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
                }
            self.print_epoch_stats(**data_element)
            
            # clusters_list = []
            # for cluster_label in set(GmmData['Label']):
            #     global all_label_dis_sum_train
            #     global all_label_code
            #     if cluster_label == None:break
            #     latent = torch.tensor([all_label_dis_sum_train[i] for i in range(len(all_label_code)) if all_label_code[i] == cluster_label])
            #     latent = latent.view(-1, 1).repeat(1, latent.shape[0])
            #     lc_dist_mat = (latent - latent.transpose(1, 0)).view(-1, 1)

            #     clusters_list.append(np.mean(lc_dist_mat.numpy()))
            # distence.append(tuple(clusters_list))
            # del clusters_list
            # all_label_dis_sum_train = []
            # all_label_code = []
            # save model
            #self.model.save()

            global all_label_dis_sum_train
            global all_label_dis_sum_val
            global clusters_distance_latent_train
            global clusters_distance_latent_valid
            if epoch_index == 0 or epoch_index % scale == (scale-1):
                distence.append(all_label_dis_sum_train)
                distence_mun.append(all_label_dis_sum_val)
                
                clusters_distance_latent_train = np.array(clusters_distance_latent_train)
                clusters_distance_latent_valid = np.array(clusters_distance_latent_valid)

                #compute and return pairwise distance of the latent code (train)
                pairwise_distances_train = np.zeros((clusters_distance_latent_train.shape[0], clusters_distance_latent_train.shape[0]),dtype=float)
        


                # Calculate pairwise Euclidean distances
                for i in range(clusters_distance_latent_train.shape[0]):
                    for j in range(i+1, clusters_distance_latent_train.shape[0]):  # To avoid calculating distances twice (i to j and j to i)
                        distance_train_latent = np.linalg.norm(clusters_distance_latent_train[i] - clusters_distance_latent_train[j]) #latent code is the repramiterized latent distribution
                        #distance = torch.linalg.norm(latent_code_mean[i] - latent_code_mean[j])
                        
                        
                        pairwise_distances_train[i][j] = distance_train_latent
                        pairwise_distances_train[j][i] = distance_train_latent

                #compute and return pairwise distance of the latent code (valid)
                pairwise_distances_Val = np.zeros((clusters_distance_latent_valid.shape[0], clusters_distance_latent_valid.shape[0]),dtype=float)
        


                # Calculate pairwise Euclidean distances
                for i in range(clusters_distance_latent_valid.shape[0]):
                    for j in range(i+1, clusters_distance_latent_valid.shape[0]):  # To avoid calculating distances twice (i to j and j to i)
                        distance_val_latent = np.linalg.norm(clusters_distance_latent_valid[i] - clusters_distance_latent_valid[j]) #latent code is the repramiterized latent distribution
                        #distance = torch.linalg.norm(latent_code_mean[i] - latent_code_mean[j])
                        
                        
                        pairwise_distances_Val[i][j] = distance_val_latent
                        pairwise_distances_Val[j][i] = distance_val_latent

                #compute and return correlation of upper triangles of the matrices
                upper_distance_train= distance_train[np.triu_indices_from(distance_train, k=1)]
                upper_distance_val= distance_val[np.triu_indices_from(distance_val, k=1)]

                upper_distance_clusters_latent_train= np.array(pairwise_distances_train)[np.triu_indices_from(np.array(pairwise_distances_train), k=1)]
                upper_distance_clusters_latent_valid= np.array(pairwise_distances_Val)[np.triu_indices_from(np.array(pairwise_distances_Val), k=1)]
                
                correlation.append(np.corrcoef(upper_distance_train,upper_distance_clusters_latent_train)[0,1])

                correlation_valid.append(np.corrcoef(upper_distance_val, upper_distance_clusters_latent_valid)[0,1])

            all_label_dis_sum_train = [0]*clusternums
            all_label_dis_sum_val = [0]*clusternums
            clusters_distance_latent_train = []
            clusters_distance_latent_valid = []

        return distence,distence_mun,correlation,correlation_valid
    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True, weightedLoss=False, probBins=[]):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)

            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, batch_num, train=train,
                weightedLoss=weightedLoss,
                probBins = probBins
            )

            # compute backward and step if train
            if train:
                #loss.register_hook(lambda grad: print(grad))
                loss.backward()
                # self.plot_grad_flow()
                self.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        if len(data_loader) == 0:
            mean_loss = 0
            mean_accuracy = 0
        else:
            mean_loss /= len(data_loader) 
            mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def cuda(self):
        """
        Convert the model to cuda
        """
        self.model.cuda()

    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.optimizer.zero_grad()

    def step(self):

        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
         # Clip gradients to prevent exploding gradients
         # Use clip_grad_norm_ or clip_grad_value_ depending on your needs
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.optimizer.step()

    def eval_model(self, data_loader, epoch_num):
        """
        This can contain any method to evaluate the performance of the mode
        Possibly add more things to the summary writer
        """
        pass

    def load_model(self):
        is_cpu = False if torch.cuda.is_available() else True
        self.model.load(cpu=is_cpu)
        if not is_cpu:
            self.model.cuda()
    



    @abstractmethod
    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True, weightedLoss=False, probBins = []):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param batch_num: int,
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        pass

    @abstractmethod
    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        pass

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        pass

    @staticmethod
    def print_epoch_stats(
            epoch_index,
            num_epochs,
            mean_loss_train,
            mean_accuracy_train,
            mean_loss_val,
            mean_accuracy_val
    ):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)

        #https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def mean_l1_loss_rnn(weights, targets):
        """
        Evaluates the mean l1 loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.L1Loss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_mse_loss_rnn(weights, targets):
        """
        Evaluates the mean mse loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.MSELoss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy_alt(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return float, accuracy
        """
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
        """

        :param z_dist: torch.distributions object
        :param prior_dist: torch.distributions
        :param beta: weight for kld loss
        :param c: capacity of bottleneck channel
        :return: kl divergence loss
        """
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = kld.sum(1).mean()
        kld = beta * (kld - c).abs()
        return kld

    @staticmethod
    def compute_reg_loss(z, inputs,label,latent_code_mean, latent_code_std, gamma,train, factor=1.0):
        """
        Computes the regularization loss
        """
        
        reg_loss = Trainer.reg_loss_sign(z, inputs,label, latent_code_mean, latent_code_std,train= train, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign(latent_code, sequence,label, latent_code_mean, latent_code_std,train, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """
        # compute latent distance matrix
        #all_label_dis_sum_train.extend(torch.norm(latent_code.detach(),dim=1))
        all_label_code.extend(label.numpy().reshape(-1))

        labeldistsum = []

# Initialize a matrix to store pairwise distances
        pairwise_distances = torch.zeros((latent_code.shape[0], latent_code.shape[0]),dtype=float)
        

        

# Calculate pairwise Euclidean distances
        for i in range(latent_code.shape[0]):
            for j in range(i+1, latent_code.shape[0]):  # To avoid calculating distances twice (i to j and j to i)
                distance = torch.linalg.norm(latent_code[i] - latent_code[j]) #latent code is the repramiterized latent distribution
                #distance = torch.linalg.norm(latent_code_mean[i] - latent_code_mean[j])
                
                
                pairwise_distances[i][j] = distance
                pairwise_distances[j][i] = distance
   



        global all_label_dis_sum_train
        global all_label_dis_sum_val
        global clusters_distance_latent_train
        global clusters_distance_latent_valid

        data_per_cluster_batch = []
        for labs in range(0,clusternums):
            seletedIndex = [idx for idx,i in enumerate(label.numpy().reshape(-1)) if i == labs]
            data_per_cluster_batch.append(seletedIndex)
            labeldistsum.append(sum([sum(pairwise_distances[iddx].detach().numpy()) for iddx in seletedIndex]))
        
        
        if train :
            all_label_dis_sum_train = np.sum(np.array([list(all_label_dis_sum_train), labeldistsum]), axis=0)
            clusters_distance_latent_train.extend(latent_code.detach().numpy())
        else:
            all_label_dis_sum_val = np.sum(np.array([list(all_label_dis_sum_val), labeldistsum]), axis=0)
            clusters_distance_latent_valid.extend(latent_code.detach().numpy())

        # compute attribute distance matrix
        attribute_dist_mat = Distencecs(sequence) #TODO: cs dist func between gmm
        attribute_dist_mat = torch.tensor(attribute_dist_mat)


        # compute regularization loss
        # loss_fn = torch.nn.L1Loss()

        lc_tanh = pairwise_distances
        attribute_sign = attribute_dist_mat *factor
        #sign_loss = loss_fn(lc_tanh, attribute_sign.float())
        sign_loss = 1 - (torch.corrcoef(torch.stack((lc_tanh.view(-1), attribute_sign.float().view(-1)),dim=0))[1,0])


        if torch.isnan(sign_loss):
            raise ValueError('sign_loss is nan')  
        
        return sign_loss

    @staticmethod
    def compute_reg_loss_weighted(self, z,inputs, labels, reg_dim, gamma, alpha, factor=1.0, probBins=[]):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign_weighted(x, inputs, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign_weighted(latent_code,sequence, factor=1.0):
        """
        Computes the regularization loss given the latent code and attribute
        Args:
            latent_code: torch Variable, (N,)
            attribute: torch Variable, (N,)
            factor: parameter for scaling the loss
        Returns
            scalar, loss
        """

     

        # set the diagonal elements to zero to remove pi * pi in sum
        # normalizing factor

         # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        # attribute = Distencecs(attribute)#.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = Distencecs(sequence)
        attribute_dist_mat = torch.tensor(attribute_dist_mat).reshape(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss(reduction = "none")
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        #{ln}= { ∣xn−yn∣ }
        elementwise_L1loss = loss_fn(lc_tanh, attribute_sign.float())

        # multiply by weights
        # elementwise_Weighted_loss = torch.mul(weights, elementwise_L1loss)

        # # sum all weighted values
        # total_weighted_loss = torch.sum(elementwise_Weighted_loss)

        # # normalize by S
        # norm_weighted_loss = total_weighted_loss / S

        return torch.mean(elementwise_L1loss)#norm_weighted_loss

    
    

    @staticmethod
    def get_save_dir(model, sub_dir_name='results'):
        path = os.path.join(
            os.path.dirname(model.filepath),
            sub_dir_name
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path
