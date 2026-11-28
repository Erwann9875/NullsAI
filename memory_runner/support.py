import ctypes
import os

import torch

def model_points(out):
    tiles = torch.arange(576)
    xy = torch.stack(((tiles % 32 + .5) / 32, (tiles // 32 + .5) / 18), dim=1)
    if out.get('offset') is not None:
        xy = (xy + out['offset'][0].detach().cpu() * torch.tensor([.5 / 32, .5 / 18])).clamp(0, 1)
    return torch.round(xy * torch.tensor([18000., 32000.])).long().numpy()


def limit_cpu():
    if os.name != 'nt':
        return dict(priority='default', affinity=None)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel.GetCurrentProcess()
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system)):
        raise OSError('Cannot read CPU affinity')
    selected = [1 << n for n in range(64) if allowed.value & (1 << n)][-2:]
    if not kernel.SetPriorityClass(handle, 0x4000) or not kernel.SetProcessAffinityMask(handle, sum(selected)):
        raise OSError('Cannot apply low-impact CPU limits')
    return dict(priority='below-normal', affinity_mask=sum(selected), logical_cpus=len(selected))
