import torch.nn.functional as F
import torchvision.transforms as T


def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss
    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape
        a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)
        a_map = a_map.sum(1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        anomaly_map += a_map
    transform = T.GaussianBlur(kernel_size=(17, 17), sigma=6.0)
    anomaly_map = transform(anomaly_map)
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    return anomaly_map






