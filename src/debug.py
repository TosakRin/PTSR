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
command = 'python main.py --data_name Beauty --do_test --do_eval --scheduler warmup+multistep --milestones "[25, 50]" --gamma 0.1 --warm_up_epochs 5 --loader_type new --gcn_mode batch --gpu_id 0 --log_root logs --gnn_layer 4 --msg training'

# Convert the command
converted_list = convert_command_to_list(command)

# Print the converted list
print(json.dumps(converted_list))
