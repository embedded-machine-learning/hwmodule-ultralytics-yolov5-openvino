#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "=== Init task spooler ==="
echo "Setup task spooler socket."

# Setup of task spooler through this repository
export TS_SOCKET="/srv/ts_socket/CPU.socket"
chmod 777 /srv/ts_socket/CPU.socket
export TS_TMPDIR=/home/intel-nuc/logs
echo task spooler output directory: ~/logs

# An alternative is to use the task spooler script of the device. Then execute that script instead
#. ../init_eda_ts.sh

echo "Task spooler initialized $TS_SOCKET"
