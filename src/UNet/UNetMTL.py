import torch
import torchvision.models as models

# Load the pretrained ResNet-50 model
resnet = models.resnet50(pretrained=True)

# Get the last convolutional layer
last_conv_layer = resnet.layer4[-1].conv3

# Create a random input tensor of the desired size
input_tensor = torch.randn(1, 3, 1536, 1536)  # Example size: (batch_size, channels, height, width)

# Pass the input through the layers up to the last convolutional layer
output = resnet.conv1(input_tensor)
output = resnet.bn1(output)
output = resnet.relu(output)
output = resnet.maxpool(output)
output = resnet.layer1(output)
output = resnet.layer2(output)
output = resnet.layer3(output)
output = resnet.layer4[0](output)
output = resnet.layer4[1](output)
output = resnet.layer4[2](output)
output = last_conv_layer(output)

# Print the dimensions of the output tensor
print(output.shape)

