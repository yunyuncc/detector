import pytest
import torch

def test_index_select():
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    inds = torch.tensor([0,2])
    
    selected = torch.index_select(x,0,inds)
    res = torch.tensor([[1,2,3],[7,8,9]])
    assert torch.all(torch.eq(selected, res)).item() == 1

    selected = torch.index_select(x,1,inds)
    res = torch.tensor([[1,3],[4,6],[7,9]])
    assert torch.all(torch.eq(selected, res)).item() == 1
def test_mask():
    x = torch.tensor([-1,2, 101, 100])
    print(x)
    mask = x < 0

    right_mask = torch.tensor([1,0,0,0], dtype=torch.uint8)
    assert(torch.all(torch.eq(right_mask, mask)))
    x[mask] = 0
    assert(torch.all(torch.eq(x, torch.tensor([0,2,101, 100]))))