import pytest
from mmdet.ops.nms.nms_wrapper import get_nms_from_type


def test_get_nms_op_for_unsupported_type():
    nms_type = 'definitely_not_nms_type'
    with pytest.raises(RuntimeError):
        get_nms_from_type(nms_type)


@pytest.mark.parametrize('supported_type', ['nms'])
def test_get_nms_op_for_supported_type(supported_type):
    nms_op = get_nms_from_type(supported_type)
    assert nms_op is not None, f'No operation found for type {supported_type}.'
