from ote_sdk.algo_backends.test_helpers.pytest_insertions import *

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name='ote/training_extensions/external/mmdetection')

# pytest magic
def pytest_generate_tests(metafunc):
    ote_pytest_generate_tests_insertion(metafunc)

def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)
