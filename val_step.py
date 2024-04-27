from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan
import torch

def val_step(val_data_iter, opt_model, criterion, criterion_weight, steps, epoch, writer, device):
    opt_model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_data_iter):
            data['mask'] = check_and_replace_nan(data['mask'])
            # mask = mask_to_one_hot(data['mask'][:,0,:,:]).to(device)
            mask = mask_to_multiclass(data['mask'], num_classes=3).to(device)
            # mask = data['mask'].to(device)
            gt_image = data['image'].to(device)
            output, output_mask, featuemaps = opt_model(data['input_img_1'].to(device), data['input_img_2'].to(device), gt_image, mask)
            if criterion_weight == None:
                criterion_weight = [1/len(criterion)]*len(criterion)
            loss = sum(map(lambda f, cw: f(output, gt_image)*cw, criterion, criterion_weight))
            # loss = criterion_weight[0]*criterion[0](output, gt_image) + criterion_weight[1]*criterion[1](output, gt_image)
            val_loss += loss.item()
            if i % steps == 0 and i != 0: break


    val_visual = torch.stack([output[0], gt_image[0], output_mask[0], mask[0]], dim=0)
    writer.add_images('val image and predicted images', val_visual, epoch)
    
    # for i in range(len(featuemaps[0])):
    # writer.add_images(f'feature map {0}', featuemaps[0][0:1,0:1], epoch)
    
    # for i in range(len(featuemaps[1])):
    #     writer.add_images(f'feature map {i}', featuemaps[1][i][0], epoch)
    
    # writer.add_images('fused features', featuemaps[2][0], epoch)
    
    return val_loss/steps