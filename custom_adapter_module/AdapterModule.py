import torch
from torch import nn
import os

class AdapterModule(nn.Module):
    """
    AdapterModule is a custom neural network module designed to adapt sentence embeddings.
    It consists of two fully connected layers followed by a dropout layer and a residual connection.
    
    Attributes:
        dense1 (nn.Linear): First fully connected layer.
        dense2 (nn.Linear): Second fully connected layer.
        output (nn.Linear): Output layer.
        activation (nn.ReLU): Activation function.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        add_residual (bool): Flag to indicate whether to add a residual connection.
        residual_weight (nn.Parameter): Weight for the residual connection.
    """
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.3, add_residual=True):
        """
        Initializes the AdapterModule with the given dimensions and dropout rate.
        
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            dropout_rate (float): Dropout rate for the dropout layer.
            add_residual (bool): Flag to indicate whether to add a residual connection. Defaults to True.
        """
        super(AdapterModule, self).__init__()
        
        # Define the first dense layer
        self.dense1 = nn.Linear(in_features=input_dim, out_features=1024, bias=True)
        
        # Define the second dense layer
        self.dense2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        
        # Define the output layer
        self.output = nn.Linear(in_features=512, out_features=output_dim)
        
        # Define the activation function
        self.activation = nn.ReLU()
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store the flag for residual connection
        self.add_residual = add_residual
        
        # Initialize the residual weight if residual connection is enabled
        if add_residual:
            self.residual_weight = nn.Parameter(nn.init.uniform_(torch.empty(1), 0, 0.1))  # Small initialization

    def forward(self, features):
        """
        Forward pass through the network.
        
        Args:
            features (dict): A dictionary containing the input data with 'sentence_embedding' as a key.
        
        Returns:
            dict: Updated input_data with modified 'sentence_embedding'.
        """
        
        # Extract the sentence embeddings from input_data
        x = features.get('sentence_embedding')
        
        # Store the original embeddings for the residual connection
        original_x = x if self.add_residual else None
        
        # Pass through the first dense layer, activation function, and dropout
        x = self.dropout(self.activation(self.dense1(x)))
        
        # Pass through the second dense layer, activation function, and dropout
        x = self.dropout(self.activation(self.dense2(x)))
        
        # Pass through the output layer
        x = self.output(x)
        
        # Add the residual connection if enabled
        if self.add_residual:
            x += self.residual_weight * original_x
            
        # Update the 'sentence_embedding' in the input_data
        features['sentence_embedding'] = x
        
        return features
    
    def save(self, output_path):
        """
        Saves the state of the model and its configuration to the specified output path.
        
        Args:
            output_path (str): Path to save the model and configuration.
        """
        
        # Save the model state dictionary
        torch.save(self.state_dict(), os.path.join(output_path, 'adapter_module.pt'))
        
        # Save the configuration of the model
        config = {
            'input_dim': self.dense1.in_features,
            'output_dim': self.output.out_features,
            'dropout_rate': self.dropout.p,
            'add_residual': self.add_residual
        }
        torch.save(config, os.path.join(output_path, 'config.pt'))
        
    @classmethod
    def load(cls, input_path):
        """
        Loads the model state and configuration from the specified input path.
        
        Args:
            input_path (str): Path to load the model and configuration from.
        
        Returns:
            AdapterModule: An instance of the AdapterModule with the loaded state and configuration.
        """
        
        # Load the configuration
        config = torch.load(os.path.join(input_path, 'config.pt'))
        
        # Create a new instance of AdapterModule with the loaded configuration
        model = cls(**config)
        
        # Load the state dictionary into the model
        model.load_state_dict(torch.load(os.path.join(input_path, 'adapter_module.pt')))
        
        return model
