# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:37

@author: mick.yi

"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.sigma = 0.25
        self.omega = 100

    def _get_features_hook(self, module, input, output):
        self.feature = output
        #print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        conv_out_conv2,conv_out_3a,conv_out_3b,conv_out_3c,conv_out_4a,conv_out_4b,\
            conv_out_4c,conv_out_4d,conv_out_4e,conv_out_5a,conv_out_5b,video_semantic,output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        if np.max(cam)!=0:
            cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))


        
        #Generate the soft-mask

        #cam_min = np.min(cam)
        #cam_max = np.max(cam)
        #scaled_cam = (cam - cam_min) / (cam_max - cam_min)
        #mask = 1/(1+np.exp(-self.omega * (cam- self.sigma)))
        #masked_image = inputs.cpu().data - inputs.cpu().data * mask

        return cam #, masked_image


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[1][0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        if np.max(cam)!=0:
            cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam

class GradCAM_two_one(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name1,layer_name2):
        self.net = net
        self.layer_name1 = layer_name1
        self.layer_name2 = layer_name2
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.sigma = 0.25
        self.omega = 100

    def _get_features_hook1(self, module, input, output):
        self.feature1 = output
        #print("feature shape:{}".format(output.size()))
    def _get_features_hook2(self, module, input, output):
        self.feature2 = output
        #print("feature shape:{}".format(output.size()))

    def _get_grads_hook1(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient1 = output_grad[0]
    def _get_grads_hook2(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient2 = output_grad[0]
    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name1:
                self.handlers.append(module.register_forward_hook(self._get_features_hook1))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook1))
            if name == self.layer_name2:
                self.handlers.append(module.register_forward_hook(self._get_features_hook2))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook2))    

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs_one, inputs_two, index_one,index_two):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output1, output2= self.net(inputs_one,inputs_two)  # [1,num_classes]
        if index_one is None:
            index_one = np.argmax(output1.cpu().data.numpy())
        #if index_two is None:
        #    index_two = np.argmax(output2.cpu().data.numpy())

        target1 = output1[0][index_one]
        #target2 = output2[0][index_two]
        target1.backward()
        #target2.backward()

        gradient1 = self.gradient1[0].cpu().data.numpy()  # [C,H,W]
        #gradient2 = self.gradient2[0].cpu().data.numpy()  # [C,H,W]

        weight1 = np.mean(gradient1, axis=(1, 2))  # [C]
        #weight2 = np.mean(gradient2, axis=(1, 2))  # [C]

        feature1 = self.feature1[0].cpu().data.numpy()  # [C,H,W]
        #feature2 = self.feature2[0].cpu().data.numpy()  # [C,H,W]

        cam1 = feature1 * weight1[:, np.newaxis, np.newaxis]  # [C,H,W]
        #cam2 = feature2 * weight2[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam1 = np.sum(cam1, axis=0)  # [H,W]
        cam1 = np.maximum(cam1, 0)  # ReLU
        #cam2 = np.sum(cam2, axis=0)  # [H,W]
        #cam2 = np.maximum(cam2, 0)  # ReLU
        # 数值归一化
        cam1 -= np.min(cam1)
        #cam2 -= np.min(cam2)
        if np.max(cam1)!=0:
            cam1 /= np.max(cam1)
        #if np.max(cam2)!=0:
        #    cam2 /= np.max(cam2)
        # resize to 224*224
        cam1 = cv2.resize(cam1, (7, 7))
        #cam2 = cv2.resize(cam2, (224, 224))

        
        #Generate the soft-mask

        #cam_min = np.min(cam)
        #cam_max = np.max(cam)
        #scaled_cam = (cam - cam_min) / (cam_max - cam_min)
        #mask = 1/(1+np.exp(-self.omega * (cam- self.sigma)))
        #masked_image = inputs.cpu().data - inputs.cpu().data * mask

        return cam1 #, masked_image

class GradCAM_two_two(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name1,layer_name2):
        self.net = net
        self.layer_name1 = layer_name1
        self.layer_name2 = layer_name2
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.sigma = 0.25
        self.omega = 100

    def _get_features_hook1(self, module, input, output):
        self.feature1 = output
        #print("feature shape:{}".format(output.size()))
    def _get_features_hook2(self, module, input, output):
        self.feature2 = output
        #print("feature shape:{}".format(output.size()))

    def _get_grads_hook1(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient1 = output_grad[0]
    def _get_grads_hook2(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient2 = output_grad[0]
    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name1:
                self.handlers.append(module.register_forward_hook(self._get_features_hook1))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook1))
            if name == self.layer_name2:
                self.handlers.append(module.register_forward_hook(self._get_features_hook2))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook2))    

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs_one, inputs_two, index_one,index_two):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output1, output2= self.net(inputs_one,inputs_two)  # [1,num_classes]
        #if index_one is None:
        #    index_one = np.argmax(output1.cpu().data.numpy())
        if index_two is None:
            index_two = np.argmax(output2.cpu().data.numpy())

        #target1 = output1[0][index_one]
        target2 = output2[0][index_two]
        #target1.backward()
        target2.backward()

        #gradient1 = self.gradient1[0].cpu().data.numpy()  # [C,H,W]
        gradient2 = self.gradient2[0].cpu().data.numpy()  # [C,H,W]

        #weight1 = np.mean(gradient1, axis=(1, 2))  # [C]
        weight2 = np.mean(gradient2, axis=(1, 2))  # [C]

        #feature1 = self.feature1[0].cpu().data.numpy()  # [C,H,W]
        feature2 = self.feature2[0].cpu().data.numpy()  # [C,H,W]

        #cam1 = feature1 * weight1[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam2 = feature2 * weight2[:, np.newaxis, np.newaxis]  # [C,H,W]
        #cam1 = np.sum(cam1, axis=0)  # [H,W]
        #cam1 = np.maximum(cam1, 0)  # ReLU
        cam2 = np.sum(cam2, axis=0)  # [H,W]
        cam2 = np.maximum(cam2, 0)  # ReLU
        # 数值归一化
        #cam1 -= np.min(cam1)
        cam2 -= np.min(cam2)
        #if np.max(cam1)!=0:
        #    cam1 /= np.max(cam1)
        if np.max(cam2)!=0:
            cam2 /= np.max(cam2)
        # resize to 224*224
        #cam1 = cv2.resize(cam1, (224, 224))
        cam2 = cv2.resize(cam2, (7, 7))

        
        #Generate the soft-mask

        #cam_min = np.min(cam)
        #cam_max = np.max(cam)
        #scaled_cam = (cam - cam_min) / (cam_max - cam_min)
        #mask = 1/(1+np.exp(-self.omega * (cam- self.sigma)))
        #masked_image = inputs.cpu().data - inputs.cpu().data * mask

        return cam2 #, masked_image