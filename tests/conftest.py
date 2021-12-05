from ote_sdk.algo_backends.test_helpers.pytest_insertions import *
from ote_sdk.algo_backends.test_helpers.training_tests_common import REALLIFE_USECASE_CONSTANT

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name='ote/training_extensions/external/mmdetection')

@pytest.fixture
def ote_test_domain_fx():
    return 'custom-object-detection'

@pytest.fixture
def ote_test_scenario_fx(current_test_parameters_fx):
    assert isinstance(current_test_parameters_fx, dict)
    if current_test_parameters_fx.get('usecase') == REALLIFE_USECASE_CONSTANT():
        return 'performance'
    else:
        return 'integration'

@pytest.fixture(scope='session')
def ote_templates_root_dir_fx():
    import os.path as osp
    import logging
    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root = f'{root}/configs/ote/'
    logger.debug(f'overloaded ote_templates_root_dir_fx: return {root}')
    return root

# pytest magic
def pytest_generate_tests(metafunc):
    ote_pytest_generate_tests_insertion(metafunc)

def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)
