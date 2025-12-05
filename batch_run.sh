#!/usr/bin/env bash
IFS=$'\n\t'

# logs
LOG_DIR="/goku/home/ec2-user/GOKU/logs"
mkdir -p "$LOG_DIR"

# mount if not mounted
if ! mountpoint -q /goku; then
  echo "Mounting /dev/nvme1n1p1 -> /goku"
  sudo mkdir -p /goku
  sudo mount /dev/nvme1n1p1 /goku
else
  echo "/goku already mounted"
fi

# activate venv or use venv python
VENV_PY="/goku/home/ec2-user/GOKU/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  # fallback to shell activation
  source /goku/home/ec2-user/GOKU/.venv/bin/activate
  VENV_PY=$(which python)
fi

cd /goku/home/ec2-user/GOKU

# Run sequentially with logs; change -u for unbuffered output
$VENV_PY -u goku_train.py --model double_pendulum 2>&1 | tee "$LOG_DIR/goku_train_double_pendulum.log"
$VENV_PY -u lstm_train.py --model pendulum 2>&1 | tee "$LOG_DIR/lstm_train_pendulum.log"
$VENV_PY -u lstm_train.py --model cvs 2>&1 | tee "$LOG_DIR/lstm_train_cvs.log"
$VENV_PY -u lstm_train.py --model double_pendulum 2>&1 | tee "$LOG_DIR/lstm_train_double_pendulum.log"
$VENV_PY -u lstm_train.py --model pendulum_friction 2>&1 | tee "$LOG_DIR/lstm_train_pendulum_friction.log"
$VENV_PY -u latent_ode_train.py --model pendulum 2>&1 | tee "$LOG_DIR/latent_ode_pendulum.log"
$VENV_PY -u latent_ode_train.py --model cvs 2>&1 | tee "$LOG_DIR/latent_ode_cvs.log"
$VENV_PY -u latent_ode_train.py --model double_pendulum 2>&1 | tee "$LOG_DIR/latent_ode_double_pendulum.log"
$VENV_PY -u latent_ode_train.py --model pendulum_friction 2>&1 | tee "$LOG_DIR/latent_ode_pendulum_friction.log"

echo "All done"

# Use tmux (recommended for debugging):
# Start: tmux new -d -s goku_run './batch_run.sh'
# Reattach: tmux attach -t goku_run

# tmux new -d -s goku_run 'bash /goku/home/ec2-user/GOKU/batch_run.sh'