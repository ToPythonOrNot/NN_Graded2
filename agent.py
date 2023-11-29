from replay_buffer import ReplayBufferNumpy
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim
import numpy as np
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss as mean_huber_loss


# No changes needed in the Agent class

class Agent(): 
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)

        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
     
        return self._gamma

    def reset_buffer(self, buffer_size=None):
       
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
     
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):

        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
      
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
      
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col): 
        return row*self._board_size + col

# PyTorch uses a more object-oriented approach than Tensorflow 
# So we change from using TensorFlow's Model and Conv2D functions to defining a 
# - DeepQNetwork class that inherits from nn.Module in PyTorch.

# PyTorch's nn.ModuleList and nn.Sequential are used for creating lists of modules (like convolutional layers) 
# - and sequential layers (like fully connected layers with activations), respectively. 
# This is a shift from the way layers are added in TensorFlow, 
# - where each layer is typically added independently.

# The forward method explicitly defines the forward pass of the network. This method 
# - processes the input through each layer. 
# This is a contrast to TensorFlow's approach, where the model's forward pass is often 
# - implicitly defined through the construction of the model.

class DeepQNetwork(nn.Module):
    
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        conv_layers = [
            (2, 16, (3, 3)),
            (16, 32, (3, 3)),
            (32, 64, (5, 5))
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size) for in_channels, out_channels, kernel_size in conv_layers]
        )
        self.flat = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):

        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.flat(x)

        x = self.fc_layers(x)
        return x

# For DeepQLearningAgent(Agent):
# Methods removed: print_models, compare_weights, set_weights_trainable.
# These methods were more relevant in TensorFlow’s context.
# PyTorch did not require these methods due to its dynamic nature and different model handling practices.

class DeepQLearningAgent(Agent):
    # Change (PyTorch): Creates the main model and optionally a target network for stable training.
    # Reason for Change: This change reflects PyTorch’s practice of explicitly resetting or initializing models, particularly important in 
    # - reinforcement learning where models might be reinitialized throughout the training process. 
    # - PyTorch’s modular approach to neural networks encourages clear separation between model creation and other aspects of setup.
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super(DeepQLearningAgent, self).__init__(
            board_size=board_size, 
            frames=frames, 
            buffer_size=buffer_size,
            gamma=gamma, 
            n_actions=n_actions, 
            use_target_net=use_target_net,
            version=version
        )
        self.reset_models()
        self.configure_optimizer()

    
    def reset_models(self):
        # Previous: TensorFlow's approach relied on building and compiling the model, including setting up the optimizer and loss function.
        # Change: (PyTorch): Creates the main model and optionally a target network for stable training.
        # Reason for Change: This change reflects PyTorch’s practice of explicitly resetting 
        # - or initializing models, particularly important in reinforcement learning where 
        # - models might be reinitialized throughout the training process. PyTorch’s modular 
        # - approach to neural networks encourages clear separation between model creation and 
        #  - other aspects of setup.
        self._model = self.create_agent_model()
        if self._use_target_net:
            self._target_net = self.create_agent_model()
            self.sync_target_network()

    def configure_optimizer(self): # New
        # Change: Sets up the RMSprop optimizer with model parameters.
        # Reason: PyTorch requires explicit optimizer setup, separate from model definition.
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)

    def prepare_input(self, board):
        # Previous (TensorFlow): In TensorFlow, the method would typically prepare the input by 
        # - reshaping it to match the model's input requirements. TensorFlow uses a 
        # - 'channels-last' format by default.
        # Change (PyTorch): Reshapes the input if necessary and adjusts the axis order for 
        # - PyTorch's 'channels-first' format.
        # Reason for Change: PyTorch uses a different data format ('channels-first') 
        # - compared to TensorFlow. This necessitates the reordering of input dimensions 
        # - to align with PyTorch’s data handling conventions.
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = np.rollaxis(board, 3, 1)
        return self.normalize_board_input(board)

    def _get_model_outputs(self, board, model=None):
        # Previous (TensorFlow): Obtained model outputs using TensorFlow's model prediction 
        # - methods, such as predict_on_batch, directly from the model.
        # Change (PyTorch): Prepares board input, converts it to a PyTorch tensor, and gets 
        # - the model's output. Includes a check for invalid outputs.
        # Reason for Change: This change ensures the input is in the correct format for PyTorch 
        # - and adds a validation step to catch any numerical issues in the model's output, 
        # - reflecting PyTorch's explicit and direct approach to handling model predictions.
        prepared_board = torch.from_numpy(self.prepare_input(board)).float()
        model = self._model if model is None else model
        outputs = model(prepared_board)
        self.check_for_invalid_outputs(outputs)
        return outputs

    def normalize_board_input(self, board): # Similar
        # Similar functionality in TensorFlow also involved normalizing the board input to facilitate learning.
        return board.astype(np.float32) / 4.0

    def check_for_invalid_outputs(self, outputs): # New
        # New Method.
        # Reason for Change: Added as a safety check in PyTorch to ensure the model’s outputs are valid. This was crucial for the 
        # - stability and reliability of the learning process and reflects PyTorch’s emphasis on robustness and error handling.
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("Warning: NaN or inf values detected in model outputs")

    def move(self, board, legal_moves, value=None):
        # Previous (TensorFlow): In TensorFlow, the method would involve obtaining model outputs 
        # - and using them to make a decision, typically without needing to detach tensors from 
        # - the computation graph, as TensorFlow handles this differently.
        # Change (PyTorch): Obtains model outputs, detaches them from the current computation graph, 
        # - converts them to NumPy arrays, and then uses these outputs to determine the next move.
        # Reason for Change: Detaching tensors from the computation graph and converting to NumPy 
        # - arrays is specific to PyTorch’s dynamic computation graph and its flexibility in 
        # - integrating with other Python libraries. This step is indicative of PyTorch's approach 
        # - to handling model outputs, especially when they are used for decision-making outside the 
        # - training loop.
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = model_outputs.detach().numpy()
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        # Previous (TensorFlow): Created the model by dynamically constructing it layer by layer, 
        # - typically using configurations loaded from a JSON file. This approach allowed for a 
        # - flexible and configurable model architecture.
        # Change (PyTorch): Returns an instance of DeepQNetwork.
        # Reason for Change: PyTorch generally encourages defining models as separate classes that 
        # - inherit from nn.Module. This approach simplifies the model creation process, enhances 
        # - readability, and aligns with the modular and object-oriented nature of PyTorch.
        return DeepQNetwork()

    def reset_models(self): #New
        # Previous (TensorFlow): Model initialization in TensorFlow could be a part of the constructor 
        # - or a separate setup method, but it didn't typically involve a distinct 'reset' concept.
        # Change (PyTorch): Resets the main model and target network using the _agent_model method.
        # Reason for Change: This method reflects PyTorch's practice of explicitly resetting models. 
        # In reinforcement learning, it's particularly important to reinitialize models during training 
        # - to start from a clean state or update model architecture dynamically.
        self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model()
            self.update_target_net()

    def get_action_proba(self, board, values=None): # Similar
        # No changes made here.
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs -= model_outputs.max(axis=1, keepdims=True)
        model_outputs = np.exp(model_outputs)
        return model_outputs / model_outputs.sum(axis=1, keepdims=True)

    def save_model(self, file_path='', iteration=0):
        # Previous (TensorFlow): Typically involved saving the entire model or just the weights using 
        # - methods like save or save_weights, in the H5 file format.
        # Change (PyTorch): Saves the model's state dictionary using torch.save.
        # Reason for Change: This change is due to PyTorch's standard practice of saving models, where 
        # - the state dictionary (containing all model parameters) is saved, in a .pt. 
        # This method is more comprehensive as it stores all model parameters and is more aligned with PyTorch’s practices.
        assert iteration is None or isinstance(iteration, int), "iteration should be an integer"
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pt")
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pt")

    def load_model(self, file_path='', iteration=0):
        # Previous (TensorFlow): Loaded model weights directly into the model using TensorFlow's 
        # - load_weights method from an H5 file format.
        # Change (PyTorch): Loads the model's state dictionary using torch.load.
        # Reason for Change: PyTorch uses a different format .pt. for saving and loading models, which includes all 
        # model parameters, providing a comprehensive snapshot of the model.
        assert iteration is None or isinstance(iteration, int), "iteration should be an integer"
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pt"))
        if self._use_target_net:
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pt"))

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        # Previous (TensorFlow): The training process in TensorFlow involved compiling the model with a 
        # - loss function and optimizer and then fitting the model on training data using methods like 
        # - train_on_batch.
        # Change (PyTorch): Includes explicit steps for loss calculation, backpropagation, and optimizer updates.
        # Reason for Change: PyTorch's dynamic graph and imperative programming approach allows direct 
        # - control over the training process. Unlike TensorFlow's compile-and-fit pattern, PyTorch 
        # - requires manual implementation of the training loop, offering more flexibility and control.
        loss_function = nn.SmoothL1Loss()
        states, actions, rewards, next_states, terminals, legal_moves = self._buffer.sample(batch_size)

        if reward_clip:
            rewards = np.clip(rewards, -1, 1)

        model_for_next_states = self._target_net if self._use_target_net else self._model
        next_state_values = self._get_model_outputs(next_states, model_for_next_states).detach().numpy()
        discounted_future_rewards = rewards + (self._gamma * np.max(np.where(legal_moves == 1, next_state_values, -np.inf), axis=1, keepdims=True)) * (1 - terminals)
        current_state_values = self._get_model_outputs(states).detach().numpy()
        updated_values = (1 - actions) * current_state_values + actions * discounted_future_rewards
        loss = loss_function(self._model(torch.from_numpy(self.prepare_input(states)).float()), torch.from_numpy(updated_values).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_net(self): # Mostly Similar
        # Previous (TensorFlow): Updated the target network weights by setting them equal to the weights of the main model, similar to PyTorch.
        # Change (PyTorch): Synchronizes the target network's weights with the main model using PyTorch’s load_state_dict method.
        # Reason for Change: The method remains fundamentally similar, but it adapts to PyTorch's way of handling model parameters and states.
        self._target_net.load_state_dict(self._model.state_dict())

    def copy_weights_from_agent(self, agent_for_copy):
        # Previous (TensorFlow): TensorFlow also supports copying weights between models using methods like set_weights.
        # Change (PyTorch): Allows copying weights from another agent instance.
        # Reason for Change: This method is particularly useful in scenarios like training multiple agents or transferring 
        # - learned policies, a feature often utilized in advanced reinforcement learning setups. PyTorch's load_state_dict 
        # - method provides a straightforward way to copy model parameters.
        assert isinstance(agent_for_copy, DeepQLearningAgent), "The provided agent_for_copy must be an instance of DeepQLearningAgent"
        self._model.load_state_dict(agent_for_copy._model.state_dict())
        if self._use_target_net:
            self._target_net.load_state_dict(agent_for_copy._target_net.state_dict())
