import pytest
import dannet as dt


def pytest_addoption(parser):
    parser.addoption(
        '--device',
        action='store',
        default='0,0',
        help='platform_id,device_id'
    )


@pytest.fixture(scope='session')
def device(request):
    device_str = request.config.getoption('--device')
    x, y = map(int, device_str.split(','))
    return dt.Device(x, y)
