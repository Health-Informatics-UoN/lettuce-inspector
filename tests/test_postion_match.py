from evaluation.metrics import first_match_in_collection

def test_matches():
    probe = [1,2,3,4,5,6,7,8,9,10]

    assert first_match_in_collection({1}, probe) == 0
    assert first_match_in_collection({10}, probe) == 9

def test_no_match():
    assert first_match_in_collection({11}, [1,2,3,4,5,6,7,8,9,10]) == -1
