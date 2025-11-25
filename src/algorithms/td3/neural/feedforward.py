import torch
import numpy as np
import neural.noisy as no

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, noisy, activation_fun=torch.nn.ReLU(), output_activation=None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        self.noisy = noisy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_sizes = [self.input_size] + self.hidden_sizes
        if noisy:
            print("Use noisy layers")
            self.layers = torch.nn.ModuleList([ no.Noisy(i, o).to(device) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
            self.readout = no.Noisy(self.hidden_sizes[-1], self.output_size).to(device)
        else:
            print("Use normal layers")
            self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o).to(device) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
            self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size).to(device)
        self.activations = [ activation_fun for l in  self.layers ]
        
    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            layer_res = layer(x)
            x = activation_fun(layer_res)
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def set_noisy(self, noisy):
        for layer in self.layers:
            layer.activated_noise = noisy
            self.readout.activated_noise = noisy

    def reset(self):
        if self.noisy:
            for layer in self.layers:
                layer.reset()
            self.readout.reset()