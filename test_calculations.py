Traceback:
File "/mount/src/cotrends/test_calculations.py", line 81, in <module>
    cot_df = get_cot_data(limit=5)
File "/mount/src/cotrends/test_calculations.py", line 44, in get_cot_data
    results = client.get(dataset_id, limit=limit)
File "/home/adminuser/venv/lib/python3.13/site-packages/sodapy/socrata.py", line 412, in get
    response = self._perform_request(
        "get", resource, headers=headers, params=params
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/sodapy/socrata.py", line 555, in _perform_request
    utils.raise_for_status(response)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/sodapy/utils.py", line 30, in raise_for_status
    raise requests.exceptions.HTTPError(http_error_msg, response=response)
