import torch
import sys

sys.path.append('/tmp/amlt-code-download/abgoswam_mega')
print('\n'.join(sys.path))

# Replace with the path to your Megatron-LM checkpoint
# checkpoint_path = '/mnt/syntheticpipelinetrainerv1/omni_unified_v1/jobs_test/test_NN_4/ckpts/iter_0010000/mp_rank_00/model_optim_rng.pt'
checkpoint_path = '/mnt/syntheticpipelinetrainerv1/omni_unified_v1/misc/model_checkpoint_2B_gpt/model_weights.ckpt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print(checkpoint)

# # Access the state dictionary
# model_state_dict = checkpoint['model']
# # model_state_dict = checkpoint['model']['language_model']['transformer']

# # Print the layers and their sizes
# # Iterate over the state_dict items and print layer names and their sizes
for layer_name, tensor in checkpoint.items():
    if isinstance(tensor, torch.Tensor):  # Check if the item is a tensor
        print(f"Layer: {layer_name}, Size: {tuple(tensor.size())}")
    else:
        print(f"Layer: {layer_name} is not a tensor, found type: {type(tensor)}")