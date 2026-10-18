from diffusers.optimization import get_scheduler
import os
 

# Usage example:
# parameters_list = get_train_params(cond, unet, is_main_process)
def get_train_params(model, trainable_params: str = "True", added_params: str = "False", output_dir: str = "."):
    """get the parameters that need to be trained from condition `cond`."""
    parameters_list = []

    # if ('after_proj_main_block' in name) and ('geo_' in name)
    if trainable_params is not None:
        for name, param in model.named_parameters():
            if eval(trainable_params):
                parameters_list.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                parameters_list.append(param)
            else:
                if eval(added_params):
                    parameters_list.append(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    print(f"Trainable condition: {trainable_params}")
    print(f"Total trainable params: {sum(p.numel() for p in parameters_list)}")
    # check parameters
    os.makedirs(output_dir, exist_ok=True)
    freeze_path = os.path.join(output_dir, "params_freeze.txt")
    train_path = os.path.join(output_dir, "params_train.txt")
    with open(freeze_path, "w") as rec_txt1, open(train_path, "w") as rec_txt2:
        for name, para in model.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')

    return parameters_list
    

# Usage example:
# optimizer, lr_scheduler = configure_optimizer_scheduler(
#     args, accelerator.num_processes,
#     optimizer_cls, parameters_list,
# )
def configure_optimizer_scheduler(args, num_processes, optimizer_cls, parameters_list):
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * num_processes,
        num_training_steps=args.max_train_steps * num_processes,
    )
    return optimizer, lr_scheduler
