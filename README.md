# CSC401A3-GMM-Autotester

How to run the tests:

1. Copy the folder containing necessary files to your account
```
cp -r /u/cs401/A3/tests <destination of choice>
cd <path/to/the/folder>
```

2. Run the test file depending on what gmm file you implemented
  * If you used `a3_gmm.py`:
```
python3 student_tests.py --gmm <path/to/your/a3_gmm_structured.py> 
```

  * If you used `a3_gmm_structured.py`
```
python3 student_tests.py --gmm <path/to/your/a3_gmm.py> 
```
