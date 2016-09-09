# Copyright (c) 2015-2016 Antonino Ingargiola <tritemio@gmail.com>
# License: MIT
"""
Script to execute notebooks in a folder, unless their name
starts with '_' or ends with '-out'.

The executed notebooks are saved with a '-out' suffix.

Usage:

    nbrun.py notebook_folder ouput_folder

"""
from __future__ import print_function
import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_name, nb_suffix='-out', out_path='.', timeout=3600,
                 **execute_kwargs):
    """Runs a notebook and saves the output in the same notebook.

    Arguments:
        notebook_name (string): name of the notebook to be executed.
        nb_suffix (string): suffix to append to the file name of the executed
            notebook.
        timeout (int): timeout in seconds after which the execution is aborted.
        execute_kwargs (dict): additional arguments passed to
            `ExecutePreprocessor`.
        out_path (string): folder where to save the output notebook.
    """
    timestamp_cell = "**Executed:** %s\n\n**Duration:** %d seconds."

    if str(notebook_name).endswith('.ipynb'):
        notebook_name = str(notebook_name)[:-len('.ipynb')]
    nb_name_input = notebook_name + '.ipynb'
    nb_name_output = notebook_name + '%s.ipynb' % nb_suffix
    nb_name_output = os.path.join(out_path, nb_name_output)
    print('- Executing: ', nb_name_input)

    execute_kwargs_ = dict(kernel_name = 'python%d' % sys.version_info[0])
    if execute_kwargs is not None:
        execute_kwargs_.update(execute_kwargs)
    ep = ExecutePreprocessor(timeout=timeout, **execute_kwargs_)
    nb = nbformat.read(nb_name_input, as_version=4)

    start_time = time.time()
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except:
        # Execution failed, print a message then raise.
        msg = 'Error executing the notebook "%s".\n\n' % notebook_name
        msg += 'See notebook "%s" for the traceback.' % nb_name_output
        print(msg)
        raise
    else:
        # On successful execution, add timestamping cell
        duration = time.time() - start_time
        timestamp_cell = timestamp_cell % (time.ctime(start_time), duration)
        nb['cells'].insert(0, nbformat.v4.new_markdown_cell(timestamp_cell))
    finally:
        # Save the notebook even when it raises an error
        nbformat.write(nb, nb_name_output)
        print('* Output: ', nb_name_output)

if __name__ == '__main__':
    from pathlib import Path
    import sys

    path = '.'
    if len(sys.argv) > 1:
        path = sys.argv[1]
        assert os.path.isdir(path), 'Folder "%s" not found.' % path
    out_path = path
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
        assert os.path.isdir(out_path), 'Folder "%s" not found.' % out_path
    print('Executing notebooks in "%s"... ' % os.path.abspath(path))
    pathlist = list(Path(path).glob('*.ipynb'))
    for nbpath in pathlist:
        if not (nbpath.stem.endswith('-out') or nbpath.stem.startswith('_')):
            print()
            run_notebook(nbpath, out_path=out_path)
