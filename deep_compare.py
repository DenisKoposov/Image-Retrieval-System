import os
from functools import partial
import torch
from torch.utils.serialization import load_lua
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.benchmark = True

def conv2d(input, params, base, stride=1, padding=0):
    return F.conv2d(input, params[base + '.weight'], params[base + '.bias'],
                    stride, padding)


def linear(input, params, base):
    return F.linear(input, params[base + '.weight'], params[base + '.bias'])


#####################   2ch   #####################

def deepcompare_2ch(input, params):
    o = conv2d(input, params, 'conv0', stride=3)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv1')
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv2')
    o = F.relu(o).view(o.size(0), -1)
    return o
    #return linear(o, params, 'fc')


#####################   2ch2stream   #####################

def deepcompare_2ch2stream(input, params):

    def stream(input, name):
        o = conv2d(input, params, name + '.conv0')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv1')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv2')
        o = F.relu(o)
        o = conv2d(o, params, name + '.conv3')
        o = F.relu(o)
        return o.view(o.size(0), -1)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), 'fovea')
    o_retina = stream(F.pad(input, (-16,) * 4), 'retina')
    return torch.cat([o_fovea, o_retina], dim=1)
    #o = linear(torch.cat([o_fovea, o_retina], dim=1), params, 'fc0')
    #return linear(F.relu(o), params, 'fc1')


#####################   siam   #####################

def siam(patch, params):
    o = conv2d(patch, params, 'conv0', stride=3)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv1')
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv2')
    o = F.relu(o)
    return o.view(o.size(0), -1)


def deepcompare_siam(input, params):
    o = linear(torch.cat(map(partial(siam, params=params), input.split(1, dim=1)),
                         dim=1), params, 'fc0')
    return linear(F.relu(o), params, 'fc1')

#####################   siam2stream   #####################


def siam_stream(patch, params, base):
    o = conv2d(patch, params, base + '.conv0', stride=2)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, base + '.conv1')
    o = F.relu(o)
    o = conv2d(o, params, base + '.conv2')
    o = F.relu(o)
    o = conv2d(o, params, base + '.conv3')
    return o.view(o.size(0), -1)


def streams(patch, params):
    o_retina = siam_stream(F.pad(patch, (-16,) * 4), params, 'retina')
    o_fovea = siam_stream(F.avg_pool2d(patch, 2, 2), params, 'fovea')
    return torch.cat([o_retina, o_fovea], dim=1)


def deepcompare_siam2stream(input, params):
    embeddings = map(partial(streams, params=params), input.split(1, dim=1))
    o = linear(torch.cat(embeddings, dim=1), params, 'fc0')
    o = F.relu(o)
    o = linear(o, params, 'fc1')
    return o


models = {
    '2ch': deepcompare_2ch,
    '2ch2stream': deepcompare_2ch2stream,
    'siam': siam,
    'siam2stream': streams
}


def load_net(model, lua_model, gpu_id='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    if torch.cuda.is_available():
        # to prevent opencv from initializing CUDA in workers
        torch.randn(8).cuda()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def cast(t):
        return t.cuda() if torch.cuda.is_available() else t

    f = models[model]
    net = load_lua(lua_model)
    print(type(net))
    print(net)

    if model == '2ch':
        params = {}
        for j, i in enumerate([0, 3, 6]):
            params['conv%d.weight' % j] = net.get(i).weight
            params['conv%d.bias' % j] = net.get(i).bias
        params['fc.weight'] = net.get(9).weight
        params['fc.bias'] = net.get(9).bias
    elif model == '2ch2stream':
        params = {}
        for j, branch in enumerate(['fovea', 'retina']):
            for k, layer in enumerate(map(net.get(0).get(j).get(1).get, [1, 4, 7, 9])):
                params['%s.conv%d.weight' % (branch, k)] = layer.weight
                params['%s.conv%d.bias' % (branch, k)] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias
    elif model == 'siam':
        params = {}
        for k, layer in enumerate(map(net.get(0).get(0).get, [1, 4, 7])):
            params['conv%d.weight' % k] = layer.weight
            params['conv%d.bias' % k] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias
    elif model == 'siam2stream':
        params = {}
        for stream, name in zip(net.get(0).get(0).modules, ['retina', 'fovea']):
            for k, layer in enumerate(map(stream.get, [2, 5, 7, 9])):
                params['%s.conv%d.weight' % (name, k)] = layer.weight
                params['%s.conv%d.bias' % (name, k)] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias

    params = {k: Variable(cast(v)) for k, v in params.items()}

    return f, params
