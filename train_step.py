from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan

def train_step(data, opt_model, criterion, criterion_weight, optimizer, grad_acc, start_step, writer, device):
    """
    Perform a single training step for the image fusion model.

    Args:
        train_data_iter (iterable): An iterable containing the training data.
        opt_model (nn.Module): The image fusion model to be trained.
        criterion (list): A list of loss functions to compute the training loss.
        criterion_weight (float or None): The weight to be applied to each loss function.
            If None, equal weight is applied to all loss functions.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        grad_acc (int): The number of gradient accumulation steps before performing an optimization step.
        start_step (int): The starting step of the training process.
        device (torch.device): The device on which the model and data tensors are located.

    Returns:
        tuple: A tuple containing the updated model, optimizer, and the total training loss.

    """
    train_loss = 0
    val_loss = 0
    opt_model.train()
    
    data['mask'] = check_and_replace_nan(data['mask'])
    # mask = mask_to_one_hot(data['mask'][:,0,:,:]).to(device)
    mask = mask_to_multiclass(data['mask'], num_classes=3).to(device)
    # mask = data['mask'].to(device)
    gt_image = data['image'].to(device)
    image1 = data['input_img_1'].to(device)
    image2 = data['input_img_2'].to(device)
    # optimizer.zero_grad()
    output, output_mask, _ = opt_model(image1, image2, gt_image, mask)
    if criterion_weight == None:
        criterion_weight = [1/len(criterion)]*len(criterion)
    loss = sum(map(lambda f, cw: f(output, gt_image)*cw, criterion, criterion_weight))
    loss = loss/grad_acc

    loss.backward()
    # optimizer.step()
    train_loss += loss.item()

    return opt_model, optimizer, train_loss