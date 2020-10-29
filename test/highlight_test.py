# 
#   Deep Edit
#   Copyright (c) 2020 Homedeck, LLC.
#

from pytest import fixture, mark
from .utility import tensorread, tensorwrite

from highlight import HighlightLoss

def test_highlight_loss ():
    image_path = "test/media/1.jpg"
    image = tensorread(image_path, size=None)
    highlight_loss = HighlightLoss()
    loss = highlight_loss(image)
    print(f"Loss: {loss}")
    assert loss > 0.