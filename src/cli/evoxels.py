import torch
import math
import bisect


def events_to_voxel_torch(
    xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True
):
    if device is None:
        device = xs.device
    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)
    bins = []
    dt = ts[-1] - ts[0]
    t_norm = (ts - ts[0]) / dt * (B - 1)
    zeros = torch.zeros_like(t_norm)
    for bi in range(B):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image_torch(
            xs,
            ys,
            weights,
            device,
            sensor_size=sensor_size,
            clip_out_of_range=False,
        )
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_image_torch(
    xs,
    ys,
    ps,
    device=None,
    sensor_size=(180, 240),
    clip_out_of_range=True,
    interpolation=None,
    padding=True,
    default=0,
):
    if device is None:
        device = xs.device
    if interpolation == "bilinear" and padding:
        img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.0], device=device)
        ones_v = torch.tensor([1.0], device=device)
        clipx = (
            img_size[1] if interpolation is None and not padding else img_size[1] - 1
        )
        clipy = (
            img_size[0] if interpolation is None and not padding else img_size[0] - 1
        )
        mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(
            ys >= clipy, zero_v, ones_v
        )

    img = (torch.ones(img_size) * default).to(device)
    if (
        interpolation == "bilinear"
        and xs.dtype is not torch.long
        and xs.dtype is not torch.long
    ):
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs - pxs).float()
        dys = (ys - pys).float()
        pxs = (pxs * mask).long()
        pys = (pys * mask).long()
        masked_ps = ps.squeeze() * mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs * mask, ys * mask
            img.index_put_((ys, xs), ps, accumulate=True)
        except Exception as e:
            print(
                "Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                    ps.shape,
                    ys.shape,
                    xs.shape,
                    img.shape,
                    torch.max(ys),
                    torch.max(xs),
                )
            )
            raise e
    return img


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
    return img


def voxel_normalization(voxel):
    tmp = torch.zeros_like(voxel)
    if torch.equal(voxel, tmp):
        return voxel
    abs_voxel, _ = torch.sort(torch.abs(voxel).view(-1, 1).squeeze(1))
    first_non_zero_idx = torch.nonzero(abs_voxel)[0].item()
    non_zero_voxel = abs_voxel[first_non_zero_idx:]
    norm_idx = math.floor(non_zero_voxel.shape[0] * 0.98)
    ones = torch.ones_like(voxel)
    normed_voxel = torch.where(
        torch.abs(voxel) < non_zero_voxel[norm_idx],
        voxel / non_zero_voxel[norm_idx],
        voxel,
    )
    normed_voxel = torch.where(
        normed_voxel >= non_zero_voxel[norm_idx], ones, normed_voxel
    )
    normed_voxel = torch.where(
        normed_voxel <= -non_zero_voxel[norm_idx], -ones, normed_voxel
    )
    return normed_voxel


def package_bidirectional_event_voxels(
    x,
    y,
    t,
    p,
    timestamp_list,
    backward,
    bins,
    sensor_size,
):
    assert x.shape == y.shape == t.shape == p.shape

    temp = t.cpu().numpy().tolist()
    output = [
        temp[
            bisect.bisect_left(temp, timestamp_list[i]) : bisect.bisect_left(
                temp, timestamp_list[i + 1]
            )
        ]
        for i in range(len(timestamp_list) - 1)
    ]

    assert len(output) == len(timestamp_list) - 1, (
        f"len(output) is {len(output)}, but len(timestamp_list) is {len(timestamp_list)}"
    )
    sum_output = []
    sum = 0
    for i in range(len(output)):
        if len(output[i]) <= 1:
            raise ValueError(f"len(output[{i}] == 0)")
        sum += len(output[i])
        sum_output.append(sum)

    assert len(sum_output) == len(output)

    start_idx = 0
    out_voxels = []
    for voxel_idx in range(len(timestamp_list) - 1):
        end_idx = start_idx + len(output[voxel_idx])

        xs = x[start_idx:end_idx]
        ys = y[start_idx:end_idx]
        ts = t[start_idx:end_idx]
        ps = p[start_idx:end_idx]

        if backward:
            t_start = timestamp_list[voxel_idx]
            t_end = timestamp_list[voxel_idx + 1]
            xs = torch.flip(xs, dims=[0])
            ys = torch.flip(ys, dims=[0])
            ts = torch.flip(t_end - ts + t_start, dims=[0])
            ps = torch.flip(-ps, dims=[0])

        voxel = events_to_voxel_torch(
            xs, ys, ts, ps, bins, device=None, sensor_size=sensor_size
        )
        normed_voxel = voxel_normalization(voxel)

        out_voxels.append(normed_voxel)
        start_idx = end_idx

    return out_voxels
