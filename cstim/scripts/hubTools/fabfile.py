import fabric.connection
from fabric import  task
from pathlib import Path
import os

@task()
def scpANN(c:fabric.connection.Connection,x = None,y = None):
    # This tool provides a way to automatically upload and download ANNdataset
    # from the computer to a Slurm cluster.
    # c.run("echo "+str(path_input)+" is being transfered to "+path_output)

    if x is None:
        x = ""
    if y is None:
        raise Exception("output path must be provided")
    ### We start by listing all files:
    def rec_iter(input_dir :Path) -> list[Path]:
        outputs = []
        files = os.listdir(input_dir)
        for e in files:
            if os.path.isdir(input_dir/e):
                outputs += rec_iter(input_dir/e)
            else:
                outputs += [input_dir/e]
        return outputs
    all_files = rec_iter(y)

    c.put("echo "+str(len(all_files)))



