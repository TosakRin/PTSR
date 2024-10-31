# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import json


def convert_command_to_list(command):
    # Split the command by spaces to get individual elements
    elements = command.split()
    # Initialize an empty list to hold the converted command
    converted_list = []
    # Iterate through the elements
    for element in elements:
        # Check if an element is a parameter (starts with --)
        if element.startswith("--"):
            # Append the parameter as is
            converted_list.append(element)
        else:
            # For non-parameter elements, check if they need special handling
            # For example, removing ".float" from "--msg base-pad_mask的.float"
            if ".float" in element:
                # Remove ".float" and append the modified element
                converted_list.append(element.replace(".float", ""))
            else:
                # Append the element surrounded by quotes
                converted_list.append(f"{element}")
    return converted_list


# Example command
command = "python main.py --data_name Beauty --rec_weight 1 --f_neg --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.5 --gpu_id 1 --lr_adam 0.001 --epochs 1000 --do_test --scheduler warmup+multistep --milestones [25,100] --gamma 0.1 --warm_up_epochs 5 --gnn_layer 3 --loader new --gcn_mode batch --msg debug"

# Convert the command
converted_list = convert_command_to_list(command)

# Print the converted list
print(json.dumps(converted_list))
