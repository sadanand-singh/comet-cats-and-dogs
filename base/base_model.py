import logging
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def short_summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable parameters: {}".format(params))
        print("----------------------------------------------------------------")
        print(self)
        print("----------------------------------------------------------------")

    def summary(self, input_size, batch_size=-1, device="cpu"):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [
                torch.randn(2, *in_size, dtype=torch.float32, device=device)
                for in_size in input_size
            ]
        else:
            x = torch.randn(2, *input_size, dtype=torch.float32, device=device)

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.apply(register_hook)

        # make a forward pass
        self(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        print("----------------------------------------------------------------")
        format_str = "{:>20}  {:>25} {:>15}"
        line_new = format_str.format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0

        inp_format_str = "[" + str(batch_size) + ", {}" * len(input_size) + "]"
        line_new = format_str.format(
            "Input", inp_format_str.format(*input_size), "{0:,}".format(0)
        )
        print(line_new)
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = format_str.format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] is True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / (1024 ** 2.0))
        total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4.0 / (1024 ** 2.0))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
        print(self)
        print("----------------------------------------------------------------")
