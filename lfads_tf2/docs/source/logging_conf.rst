##########
Logging Configuration
##########

The `logging_conf.yaml` file specifies how logging is set up for `lfads_tf2`.
We create a logger for the LFADS output and for the training data. The 
LFADS output is logged to both the console and to a file in the model 
directory. The file logging includes time stamps, while the console logging 
doesn't. The training data is logged only to a CSV file in the model 
directory.

Please check out the documentation for the schema used in this YAML file 
[here](https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema).
