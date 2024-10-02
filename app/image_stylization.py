import cv2
from PIL import Image
import PIL
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1).to("cuda" if torch.cuda.is_available() else "cpu")
        self.std = torch.tensor(std).view(-1, 1, 1).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

    
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

    
class Style_transferer:
    def __init__(self, image_width = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if image_width == None:
            self.imsize = 512 if torch.cuda.is_available() else 128
        else:
            self.imsize = image_width
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),  # scale imported image
            transforms.ToTensor()])
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.cnn = self.cnn.to(self.device)
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


    def image_loader(self, image):
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def init_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        self.optimizer = optim.LBFGS([input_img])

    def init_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers=['conv_4'],
                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std)

        # just in order to have an iterable access to or list of content/style
        # losses
        self.content_losses = []
        self.style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        self.model = nn.Sequential(normalization)
        self.model = self.model.to(self.device)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]

    def run_style_transfer(self, content_img, style_img, input_img, num_steps=100,
                        style_weight=1000000, content_weight=1, verbose = False):
        """Run the style transfer."""
        if verbose:
            print('Building the style transfer model..')
        score = [0]
        score[0] = 1e+20
        self.init_style_model_and_losses(self.cnn,
            self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        self.model.eval()
        self.model.requires_grad_(False)

        self.init_input_optimizer(input_img)

        if verbose:
            print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                self.optimizer.zero_grad()
                self.model(input_img)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                self.loss = style_score + content_score
                self.loss.backward(),

                run[0] += 1
                if verbose:
                    if run[0] % 10 == 0:
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                            style_score.item(), content_score.item()))
                        print()
                score_this = style_score.item() + content_score.item()
                #if score_this<score[0] and run[0]>100:
                #    torch.save(self.model, "model.trch")
                #    score[0] = score_this
                return style_score + content_score

            self.optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    def stylize(self, image, style_img, num_steps=100, style_weight=1000000, content_weight=1, verbose = False):
        image = Image.open(image)
        width, height = image.size
        ratio = width / height
        resize_height = self.imsize / ratio
        image = image.resize((self.imsize, self.imsize))
        style_img = Image.open(style_img)
        style_img = style_img.resize((self.imsize, self.imsize))
        style_img = self.image_loader(style_img)
        image = self.image_loader(image)
        input_img = image.clone()
        out = self.run_style_transfer(image, style_img, input_img, num_steps, style_weight, content_weight, verbose)
        out = out.squeeze(0)
        out = out.permute(1, 2, 0).cpu().detach().numpy()
        out = np.uint8(out*255)
        out = Image.fromarray(out)
        out = out.resize((self.imsize, int(resize_height)))
        return out
    
    def stylize_from_pil(self, image, style_img, num_steps=100, style_weight=1000000, content_weight=1, verbose = False):
        width, height = image.size
        ratio = width / height
        resize_height = self.imsize / ratio
        image = image.resize((self.imsize, self.imsize))
        style_img = Image.open(style_img)
        style_img = style_img.resize((self.imsize, self.imsize))
        style_img = self.image_loader(style_img)
        image = self.image_loader(image)
        input_img = image.clone()
        out = self.run_style_transfer(image, style_img, input_img, num_steps, style_weight, content_weight, verbose)
        out = out.squeeze(0)
        out = out.permute(1, 2, 0).cpu().detach().numpy()
        out = np.uint8(out*255)
        out = Image.fromarray(out)
        out = out.resize((self.imsize, int(resize_height)))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', metavar='path', required=True,
                        help='the path to original image')
    parser.add_argument('--style_img', metavar='path', required=True,
                        help='path to style image')
    parser.add_argument('--output', metavar='path', required=False,
                        help='path to resulting image', default='stylized_output.jpg')
    parser.add_argument('--image_width', type = int, required=False,
                        help='width of output video')
    parser.add_argument('--num_steps', type = int, required=False,
                        help='num of steps to processing image', default=100)
    parser.add_argument('--style_weight', type = int, required=False,
                        help='rweight of style part for processing', default=1000000)
    parser.add_argument('--content_weight', type = int, required=False,
                        help='weight of content part for processing', default=1)
    parser.add_argument('--verbose', type = bool, required=False,
                        help='show process indicators', default=False)
    
    args = parser.parse_args()

    st = Style_transferer(args.image_width)
    out = st.stylize(args.image, args.style_img, args.num_steps, args.style_weight, args.content_weight, args.verbose)
    out.save(args.output)
    if args.verbose:
        print('Done')

if __name__ == '__main__':
    import argparse
    main()