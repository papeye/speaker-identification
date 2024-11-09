from main import main

def run_main():
    correctly, total = main()
    return correctly / total


def test_correctly_identifed():
    # TODO Once we fix testing data this number needs to be increased to at least 0.9
    assert run_main() >= 1.5