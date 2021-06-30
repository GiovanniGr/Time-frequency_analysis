# Time-frequency analysis
The repo contains the codes used in the "Time-frequency analysis of EEG 21" course.
The codes cover and support all the parts of the report, providing the results.
As explained extensively in the report, the results are not reliable and therefore the codes should still be debugged more.

Each file covers an aspect of the final analysis.
In particular, the file "tf_evoked.py" computes the powers of the evoked signals, starting from the .mat trials.
"tf_induc.py" does the same but for the induced signals.
Then, "permutation_cluster_test_evoked.py" and "permutation_cluster_test_induced.py" load the powers respectively of the evoked and the induced signals
and they perform the permutation cluster tests.
Lastly, "clusters_visualisation.py" shows the results of the tests.
