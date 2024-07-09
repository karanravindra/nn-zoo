import pytest

def test_import():
    import ml_zoo
    assert ml_zoo.__name__ == 'ml_zoo'
    assert ml_zoo.__version__ == '0.1.0'

if __name__ == '__main__':
    pytest.main()
