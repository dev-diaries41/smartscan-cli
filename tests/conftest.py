import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--images",
        nargs="+",
        help="Image paths",
    )

@pytest.fixture
def images(request):
    return request.config.getoption("images")